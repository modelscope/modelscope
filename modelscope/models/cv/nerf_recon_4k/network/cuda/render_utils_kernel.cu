#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

/*
   Points sampling helper functions.
 */
template <typename scalar_t>
__global__ void infer_t_minmax_cuda_kernel(
        scalar_t* __restrict__ rays_o,
        scalar_t* __restrict__ rays_d,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        const float near, const float far, const int n_rays,
        scalar_t* __restrict__ t_min,
        scalar_t* __restrict__ t_max) {
  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_ray<n_rays) {
    const int offset = i_ray * 3;
    float vx = ((rays_d[offset  ]==0) ? 1e-6 : rays_d[offset  ]);
    float vy = ((rays_d[offset+1]==0) ? 1e-6 : rays_d[offset+1]);
    float vz = ((rays_d[offset+2]==0) ? 1e-6 : rays_d[offset+2]);
    float ax = (xyz_max[0] - rays_o[offset  ]) / vx;
    float ay = (xyz_max[1] - rays_o[offset+1]) / vy;
    float az = (xyz_max[2] - rays_o[offset+2]) / vz;
    float bx = (xyz_min[0] - rays_o[offset  ]) / vx;
    float by = (xyz_min[1] - rays_o[offset+1]) / vy;
    float bz = (xyz_min[2] - rays_o[offset+2]) / vz;
    t_min[i_ray] = max(min(max(max(min(ax, bx), min(ay, by)), min(az, bz)), far), near);
    t_max[i_ray] = max(min(min(min(max(ax, bx), max(ay, by)), max(az, bz)), far), near);
  }
}

template <typename scalar_t>
__global__ void infer_n_samples_cuda_kernel(
        scalar_t* __restrict__ rays_d,
        scalar_t* __restrict__ t_min,
        scalar_t* __restrict__ t_max,
        const float stepdist,
        const int n_rays,
        int64_t* __restrict__ n_samples) {
  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_ray<n_rays) {
    const int offset = i_ray * 3;
    const float rnorm = sqrt(
            rays_d[offset  ]*rays_d[offset  ] +\
            rays_d[offset+1]*rays_d[offset+1] +\
            rays_d[offset+2]*rays_d[offset+2]);
    // at least 1 point for easier implementation in the later sample_pts_on_rays_cuda
    n_samples[i_ray] = max(ceil((t_max[i_ray]-t_min[i_ray]) * rnorm / stepdist), 1.);
  }
}

template <typename scalar_t>
__global__ void infer_ray_start_dir_cuda_kernel(
        scalar_t* __restrict__ rays_o,
        scalar_t* __restrict__ rays_d,
        scalar_t* __restrict__ t_min,
        const int n_rays,
        scalar_t* __restrict__ rays_start,
        scalar_t* __restrict__ rays_dir) {
  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_ray<n_rays) {
    const int offset = i_ray * 3;
    const float rnorm = sqrt(
            rays_d[offset  ]*rays_d[offset  ] +\
            rays_d[offset+1]*rays_d[offset+1] +\
            rays_d[offset+2]*rays_d[offset+2]);
    rays_start[offset  ] = rays_o[offset  ] + rays_d[offset  ] * t_min[i_ray];
    rays_start[offset+1] = rays_o[offset+1] + rays_d[offset+1] * t_min[i_ray];
    rays_start[offset+2] = rays_o[offset+2] + rays_d[offset+2] * t_min[i_ray];
    rays_dir  [offset  ] = rays_d[offset  ] / rnorm;
    rays_dir  [offset+1] = rays_d[offset+1] / rnorm;
    rays_dir  [offset+2] = rays_d[offset+2] / rnorm;
  }
}


std::vector<torch::Tensor> infer_t_minmax_cuda(
        torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor xyz_min, torch::Tensor xyz_max,
        const float near, const float far) {
  const int n_rays = rays_o.size(0);
  auto t_min = torch::empty({n_rays}, rays_o.options());
  auto t_max = torch::empty({n_rays}, rays_o.options());

  const int threads = 256;
  const int blocks = (n_rays + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(rays_o.type(), "infer_t_minmax_cuda", ([&] {
    infer_t_minmax_cuda_kernel<scalar_t><<<blocks, threads>>>(
        rays_o.data<scalar_t>(),
        rays_d.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        near, far, n_rays,
        t_min.data<scalar_t>(),
        t_max.data<scalar_t>());
  }));

  return {t_min, t_max};
}

torch::Tensor infer_n_samples_cuda(torch::Tensor rays_d, torch::Tensor t_min, torch::Tensor t_max, const float stepdist) {
  const int n_rays = t_min.size(0);
  auto n_samples = torch::empty({n_rays}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  const int threads = 256;
  const int blocks = (n_rays + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES(t_min.type(), "infer_n_samples_cuda", ([&] {
    infer_n_samples_cuda_kernel<scalar_t><<<blocks, threads>>>(
        rays_d.data<scalar_t>(),
        t_min.data<scalar_t>(),
        t_max.data<scalar_t>(),
        stepdist,
        n_rays,
        n_samples.data<int64_t>());
  }));
  return n_samples;
}

std::vector<torch::Tensor> infer_ray_start_dir_cuda(torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor t_min) {
  const int n_rays = rays_o.size(0);
  const int threads = 256;
  const int blocks = (n_rays + threads - 1) / threads;
  auto rays_start = torch::empty_like(rays_o);
  auto rays_dir = torch::empty_like(rays_o);
  AT_DISPATCH_FLOATING_TYPES(rays_o.type(), "infer_ray_start_dir_cuda", ([&] {
    infer_ray_start_dir_cuda_kernel<scalar_t><<<blocks, threads>>>(
        rays_o.data<scalar_t>(),
        rays_d.data<scalar_t>(),
        t_min.data<scalar_t>(),
        n_rays,
        rays_start.data<scalar_t>(),
        rays_dir.data<scalar_t>());
  }));
  return {rays_start, rays_dir};
}

/*
   Sampling query points on rays.
 */
__global__ void __set_1_at_ray_seg_start(
        int64_t* __restrict__ ray_id,
        int64_t* __restrict__ N_steps_cumsum,
        const int n_rays) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(0<idx && idx<n_rays) {
    ray_id[N_steps_cumsum[idx-1]] = 1;
  }
}

__global__ void __set_step_id(
        int64_t* __restrict__ step_id,
        int64_t* __restrict__ ray_id,
        int64_t* __restrict__ N_steps_cumsum,
        const int total_len) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<total_len) {
      const int rid = ray_id[idx];
      step_id[idx] = idx - ((rid!=0) ? N_steps_cumsum[rid-1] : 0);
    }
}

template <typename scalar_t>
__global__ void sample_pts_on_rays_cuda_kernel(
        scalar_t* __restrict__ rays_start,
        scalar_t* __restrict__ rays_dir,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        int64_t* __restrict__ ray_id,
        int64_t* __restrict__ step_id,
        const float stepdist, const int total_len,
        scalar_t* __restrict__ rays_pts,
        bool* __restrict__ mask_outbbox) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<total_len) {
    const int i_ray = ray_id[idx];
    const int i_step = step_id[idx];

    const int offset_p = idx * 3;
    const int offset_r = i_ray * 3;
    const float dist = stepdist * i_step;
    const float px = rays_start[offset_r  ] + rays_dir[offset_r  ] * dist;
    const float py = rays_start[offset_r+1] + rays_dir[offset_r+1] * dist;
    const float pz = rays_start[offset_r+2] + rays_dir[offset_r+2] * dist;
    rays_pts[offset_p  ] = px;
    rays_pts[offset_p+1] = py;
    rays_pts[offset_p+2] = pz;
    mask_outbbox[idx] = (xyz_min[0]>px) | (xyz_min[1]>py) | (xyz_min[2]>pz) | \
                        (xyz_max[0]<px) | (xyz_max[1]<py) | (xyz_max[2]<pz);
  }
}

std::vector<torch::Tensor> sample_pts_on_rays_cuda(
        torch::Tensor rays_o, torch::Tensor rays_d,
        torch::Tensor xyz_min, torch::Tensor xyz_max,
        const float near, const float far, const float stepdist) {
  const int threads = 256;
  const int n_rays = rays_o.size(0);

  // Compute ray-bbox intersection
  auto t_minmax = infer_t_minmax_cuda(rays_o, rays_d, xyz_min, xyz_max, near, far);
  auto t_min = t_minmax[0];
  auto t_max = t_minmax[1];

  // Compute the number of points required.
  // Assign ray index and step index to each.
  auto N_steps = infer_n_samples_cuda(rays_d, t_min, t_max, stepdist);
  auto N_steps_cumsum = N_steps.cumsum(0);
  const int total_len = N_steps.sum().item<int>();
  auto ray_id = torch::zeros({total_len}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  __set_1_at_ray_seg_start<<<(n_rays+threads-1)/threads, threads>>>(
        ray_id.data<int64_t>(), N_steps_cumsum.data<int64_t>(), n_rays);
  ray_id.cumsum_(0);
  auto step_id = torch::empty({total_len}, ray_id.options());
  __set_step_id<<<(total_len+threads-1)/threads, threads>>>(
        step_id.data<int64_t>(), ray_id.data<int64_t>(), N_steps_cumsum.data<int64_t>(), total_len);

  // Compute the global xyz of each point
  auto rays_start_dir = infer_ray_start_dir_cuda(rays_o, rays_d, t_min);
  auto rays_start = rays_start_dir[0];
  auto rays_dir = rays_start_dir[1];

  auto rays_pts = torch::empty({total_len, 3}, torch::dtype(rays_o.dtype()).device(torch::kCUDA));
  auto mask_outbbox = torch::empty({total_len}, torch::dtype(torch::kBool).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(rays_o.type(), "sample_pts_on_rays_cuda", ([&] {
    sample_pts_on_rays_cuda_kernel<scalar_t><<<(total_len+threads-1)/threads, threads>>>(
        rays_start.data<scalar_t>(),
        rays_dir.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        ray_id.data<int64_t>(),
        step_id.data<int64_t>(),
        stepdist, total_len,
        rays_pts.data<scalar_t>(),
        mask_outbbox.data<bool>());
  }));
  return {rays_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max};
}

template <typename scalar_t>
__global__ void sample_ndc_pts_on_rays_cuda_kernel(
        const scalar_t* __restrict__ rays_o,
        const scalar_t* __restrict__ rays_d,
        const scalar_t* __restrict__ xyz_min,
        const scalar_t* __restrict__ xyz_max,
        const int N_samples, const int n_rays,
        scalar_t* __restrict__ rays_pts,
        bool* __restrict__ mask_outbbox) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<N_samples*n_rays) {
    const int i_ray = idx / N_samples;
    const int i_step = idx % N_samples;

    const int offset_p = idx * 3;
    const int offset_r = i_ray * 3;
    const float dist = ((float)i_step) / (N_samples-1);
    const float px = rays_o[offset_r  ] + rays_d[offset_r  ] * dist;
    const float py = rays_o[offset_r+1] + rays_d[offset_r+1] * dist;
    const float pz = rays_o[offset_r+2] + rays_d[offset_r+2] * dist;
    rays_pts[offset_p  ] = px;
    rays_pts[offset_p+1] = py;
    rays_pts[offset_p+2] = pz;
    mask_outbbox[idx] = (xyz_min[0]>px) | (xyz_min[1]>py) | (xyz_min[2]>pz) | \
                        (xyz_max[0]<px) | (xyz_max[1]<py) | (xyz_max[2]<pz);
  }
}

std::vector<torch::Tensor> sample_ndc_pts_on_rays_cuda(
        torch::Tensor rays_o, torch::Tensor rays_d,
        torch::Tensor xyz_min, torch::Tensor xyz_max,
        const int N_samples) {
  const int threads = 256;
  const int n_rays = rays_o.size(0);

  auto rays_pts = torch::empty({n_rays, N_samples, 3}, torch::dtype(rays_o.dtype()).device(torch::kCUDA));
  auto mask_outbbox = torch::empty({n_rays, N_samples}, torch::dtype(torch::kBool).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(rays_o.type(), "sample_ndc_pts_on_rays_cuda", ([&] {
    sample_ndc_pts_on_rays_cuda_kernel<scalar_t><<<(n_rays*N_samples+threads-1)/threads, threads>>>(
        rays_o.data<scalar_t>(),
        rays_d.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        N_samples, n_rays,
        rays_pts.data<scalar_t>(),
        mask_outbbox.data<bool>());
  }));
  return {rays_pts, mask_outbbox};
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t norm3(const scalar_t x, const scalar_t y, const scalar_t z) {
  return sqrt(x*x + y*y + z*z);
}

template <typename scalar_t>
__global__ void sample_bg_pts_on_rays_cuda_kernel(
        const scalar_t* __restrict__ rays_o,
        const scalar_t* __restrict__ rays_d,
        const scalar_t* __restrict__ t_max,
        const float bg_preserve,
        const int N_samples, const int n_rays,
        scalar_t* __restrict__ rays_pts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<N_samples*n_rays) {
    const int i_ray = idx / N_samples;
    const int i_step = idx % N_samples;

    const int offset_p = idx * 3;
    const int offset_r = i_ray * 3;
    /* Original pytorch implementation
    ori_t_outer = t_max[:,None] - 1 + 1 / torch.linspace(1, 0, N_outer+1)[:-1]
    ori_ray_pts_outer = (rays_o[:,None,:] + rays_d[:,None,:] * ori_t_outer[:,:,None]).reshape(-1,3)
    t_outer = ori_ray_pts_outer.norm(dim=-1)
    R_outer = t_outer / ori_ray_pts_outer.abs().amax(1)
    # r = R * R / t
    o2i_p = R_outer.pow(2) / t_outer.pow(2) * (1-self.bg_preserve) + R_outer / t_outer * self.bg_preserve
    ray_pts_outer = (ori_ray_pts_outer * o2i_p[:,None]).reshape(len(rays_o), -1, 3)
   */
    const float t_inner = t_max[i_ray];
    const float ori_t_outer = t_inner - 1. + 1. / (1. - ((float)i_step) / N_samples);
    const float ori_ray_pts_x =  rays_o[offset_r  ] + rays_d[offset_r  ] * ori_t_outer;
    const float ori_ray_pts_y =  rays_o[offset_r+1] + rays_d[offset_r+1] * ori_t_outer;
    const float ori_ray_pts_z =  rays_o[offset_r+2] + rays_d[offset_r+2] * ori_t_outer;
    const float t_outer = norm3(ori_ray_pts_x, ori_ray_pts_y, ori_ray_pts_z);
    const float ori_ray_pts_m = max(abs(ori_ray_pts_x), max(abs(ori_ray_pts_y), abs(ori_ray_pts_z)));
    const float R_outer = t_outer / ori_ray_pts_m;
    const float o2i_p = R_outer*R_outer / (t_outer*t_outer) * (1.-bg_preserve) + R_outer / t_outer * bg_preserve;
    const float px = ori_ray_pts_x * o2i_p;
    const float py = ori_ray_pts_y * o2i_p;
    const float pz = ori_ray_pts_z * o2i_p;
    rays_pts[offset_p  ] = px;
    rays_pts[offset_p+1] = py;
    rays_pts[offset_p+2] = pz;
  }
}

torch::Tensor sample_bg_pts_on_rays_cuda(
        torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor t_max,
        const float bg_preserve, const int N_samples) {
  const int threads = 256;
  const int n_rays = rays_o.size(0);

  auto rays_pts = torch::empty({n_rays, N_samples, 3}, torch::dtype(rays_o.dtype()).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(rays_o.type(), "sample_bg_pts_on_rays_cuda", ([&] {
    sample_bg_pts_on_rays_cuda_kernel<scalar_t><<<(n_rays*N_samples+threads-1)/threads, threads>>>(
        rays_o.data<scalar_t>(),
        rays_d.data<scalar_t>(),
        t_max.data<scalar_t>(),
        bg_preserve,
        N_samples, n_rays,
        rays_pts.data<scalar_t>());
  }));
  return rays_pts;
}


/*
   MaskCache lookup to skip known freespace.
 */

static __forceinline__ __device__
bool check_xyz(int i, int j, int k, int sz_i, int sz_j, int sz_k) {
  return (0 <= i) && (i < sz_i) && (0 <= j) && (j < sz_j) && (0 <= k) && (k < sz_k);
}


template <typename scalar_t>
__global__ void maskcache_lookup_cuda_kernel(
    bool* __restrict__ world,
    scalar_t* __restrict__ xyz,
    bool* __restrict__ out,
    scalar_t* __restrict__ xyz2ijk_scale,
    scalar_t* __restrict__ xyz2ijk_shift,
    const int sz_i, const int sz_j, const int sz_k, const int n_pts) {

  const int i_pt = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_pt<n_pts) {
    const int offset = i_pt * 3;
    const int i = round(xyz[offset  ] * xyz2ijk_scale[0] + xyz2ijk_shift[0]);
    const int j = round(xyz[offset+1] * xyz2ijk_scale[1] + xyz2ijk_shift[1]);
    const int k = round(xyz[offset+2] * xyz2ijk_scale[2] + xyz2ijk_shift[2]);
    if(check_xyz(i, j, k, sz_i, sz_j, sz_k)) {
      out[i_pt] = world[i*sz_j*sz_k + j*sz_k + k];
    }
  }
}

torch::Tensor maskcache_lookup_cuda(
        torch::Tensor world,
        torch::Tensor xyz,
        torch::Tensor xyz2ijk_scale,
        torch::Tensor xyz2ijk_shift) {

  const int sz_i = world.size(0);
  const int sz_j = world.size(1);
  const int sz_k = world.size(2);
  const int n_pts = xyz.size(0);

  auto out = torch::zeros({n_pts}, torch::dtype(torch::kBool).device(torch::kCUDA));
  if(n_pts==0) {
    return out;
  }

  const int threads = 256;
  const int blocks = (n_pts + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(xyz.type(), "maskcache_lookup_cuda", ([&] {
    maskcache_lookup_cuda_kernel<scalar_t><<<blocks, threads>>>(
        world.data<bool>(),
        xyz.data<scalar_t>(),
        out.data<bool>(),
        xyz2ijk_scale.data<scalar_t>(),
        xyz2ijk_shift.data<scalar_t>(),
        sz_i, sz_j, sz_k, n_pts);
  }));

  return out;
}


/*
    Ray marching helper function.
 */
template <typename scalar_t>
__global__ void raw2alpha_cuda_kernel(
    scalar_t* __restrict__ density,
    const float shift, const float interval, const int n_pts,
    scalar_t* __restrict__ exp_d,
    scalar_t* __restrict__ alpha) {

  const int i_pt = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_pt<n_pts) {
    const scalar_t e = exp(density[i_pt] + shift); // can be inf
    exp_d[i_pt] = e;
    alpha[i_pt] = 1 - pow(1 + e, -interval);
  }
}

template <typename scalar_t>
__global__ void raw2alpha_nonuni_cuda_kernel(
    scalar_t* __restrict__ density,
    const float shift, scalar_t* __restrict__ interval, const int n_pts,
    scalar_t* __restrict__ exp_d,
    scalar_t* __restrict__ alpha) {

  const int i_pt = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_pt<n_pts) {
    const scalar_t e = exp(density[i_pt] + shift); // can be inf
    exp_d[i_pt] = e;
    alpha[i_pt] = 1 - pow(1 + e, -interval[i_pt]);
  }
}

std::vector<torch::Tensor> raw2alpha_cuda(torch::Tensor density, const float shift, const float interval) {

  const int n_pts = density.size(0);
  auto exp_d = torch::empty_like(density);
  auto alpha = torch::empty_like(density);
  if(n_pts==0) {
    return {exp_d, alpha};
  }

  const int threads = 256;
  const int blocks = (n_pts + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(density.type(), "raw2alpha_cuda", ([&] {
    raw2alpha_cuda_kernel<scalar_t><<<blocks, threads>>>(
        density.data<scalar_t>(),
        shift, interval, n_pts,
        exp_d.data<scalar_t>(),
        alpha.data<scalar_t>());
  }));

  return {exp_d, alpha};
}

std::vector<torch::Tensor> raw2alpha_nonuni_cuda(torch::Tensor density, const float shift, torch::Tensor interval) {

  const int n_pts = density.size(0);
  auto exp_d = torch::empty_like(density);
  auto alpha = torch::empty_like(density);
  if(n_pts==0) {
    return {exp_d, alpha};
  }

  const int threads = 256;
  const int blocks = (n_pts + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(density.type(), "raw2alpha_cuda", ([&] {
    raw2alpha_nonuni_cuda_kernel<scalar_t><<<blocks, threads>>>(
        density.data<scalar_t>(),
        shift, interval.data<scalar_t>(), n_pts,
        exp_d.data<scalar_t>(),
        alpha.data<scalar_t>());
  }));

  return {exp_d, alpha};
}

template <typename scalar_t>
__global__ void raw2alpha_backward_cuda_kernel(
    scalar_t* __restrict__ exp_d,
    scalar_t* __restrict__ grad_back,
    const float interval, const int n_pts,
    scalar_t* __restrict__ grad) {

  const int i_pt = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_pt<n_pts) {
    grad[i_pt] = min(exp_d[i_pt], 1e10) * pow(1+exp_d[i_pt], -interval-1) * interval * grad_back[i_pt];
  }
}

template <typename scalar_t>
__global__ void raw2alpha_nonuni_backward_cuda_kernel(
    scalar_t* __restrict__ exp_d,
    scalar_t* __restrict__ grad_back,
    scalar_t* __restrict__ interval, const int n_pts,
    scalar_t* __restrict__ grad) {

  const int i_pt = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_pt<n_pts) {
    grad[i_pt] = min(exp_d[i_pt], 1e10) * pow(1+exp_d[i_pt], -interval[i_pt]-1) * interval[i_pt] * grad_back[i_pt];
  }
}

torch::Tensor raw2alpha_backward_cuda(torch::Tensor exp_d, torch::Tensor grad_back, const float interval) {

  const int n_pts = exp_d.size(0);
  auto grad = torch::empty_like(exp_d);
  if(n_pts==0) {
    return grad;
  }

  const int threads = 256;
  const int blocks = (n_pts + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(exp_d.type(), "raw2alpha_backward_cuda", ([&] {
    raw2alpha_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
        exp_d.data<scalar_t>(),
        grad_back.data<scalar_t>(),
        interval, n_pts,
        grad.data<scalar_t>());
  }));

  return grad;
}

torch::Tensor raw2alpha_nonuni_backward_cuda(torch::Tensor exp_d, torch::Tensor grad_back, torch::Tensor interval) {

  const int n_pts = exp_d.size(0);
  auto grad = torch::empty_like(exp_d);
  if(n_pts==0) {
    return grad;
  }

  const int threads = 256;
  const int blocks = (n_pts + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(exp_d.type(), "raw2alpha_backward_cuda", ([&] {
    raw2alpha_nonuni_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
        exp_d.data<scalar_t>(),
        grad_back.data<scalar_t>(),
        interval.data<scalar_t>(), n_pts,
        grad.data<scalar_t>());
  }));

  return grad;
}

template <typename scalar_t>
__global__ void alpha2weight_cuda_kernel(
    scalar_t* __restrict__ alpha,
    const int n_rays,
    scalar_t* __restrict__ weight,
    scalar_t* __restrict__ T,
    scalar_t* __restrict__ alphainv_last,
    int64_t* __restrict__ i_start,
    int64_t* __restrict__ i_end) {

  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_ray<n_rays) {
    const int i_s = i_start[i_ray];
    const int i_e_max = i_end[i_ray];

    float T_cum = 1.;
    int i;
    for(i=i_s; i<i_e_max; ++i) {
      T[i] = T_cum;
      weight[i] = T_cum * alpha[i];
      T_cum *= (1. - alpha[i]);
      if(T_cum<1e-3) {
        i+=1;
        break;
      }
    }
    i_end[i_ray] = i;
    alphainv_last[i_ray] = T_cum;
  }
}

__global__ void __set_i_for_segment_start_end(
        int64_t* __restrict__ ray_id,
        const int n_pts,
        int64_t* __restrict__ i_start,
        int64_t* __restrict__ i_end) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(0<index && index<n_pts && ray_id[index]!=ray_id[index-1]) {
    i_start[ray_id[index]] = index;
    i_end[ray_id[index-1]] = index;
  }
}

std::vector<torch::Tensor> alpha2weight_cuda(torch::Tensor alpha, torch::Tensor ray_id, const int n_rays) {

  const int n_pts = alpha.size(0);
  const int threads = 256;

  auto weight = torch::zeros_like(alpha);
  auto T = torch::ones_like(alpha);
  auto alphainv_last = torch::ones({n_rays}, alpha.options());
  auto i_start = torch::zeros({n_rays}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto i_end = torch::zeros({n_rays}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  if(n_pts==0) {
    return {weight, T, alphainv_last, i_start, i_end};
  }

  __set_i_for_segment_start_end<<<(n_pts+threads-1)/threads, threads>>>(
          ray_id.data<int64_t>(), n_pts, i_start.data<int64_t>(), i_end.data<int64_t>());
  i_end[ray_id[n_pts-1]] = n_pts;

  const int blocks = (n_rays + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(alpha.type(), "alpha2weight_cuda", ([&] {
    alpha2weight_cuda_kernel<scalar_t><<<blocks, threads>>>(
        alpha.data<scalar_t>(),
        n_rays,
        weight.data<scalar_t>(),
        T.data<scalar_t>(),
        alphainv_last.data<scalar_t>(),
        i_start.data<int64_t>(),
        i_end.data<int64_t>());
  }));

  return {weight, T, alphainv_last, i_start, i_end};
}

template <typename scalar_t>
__global__ void alpha2weight_backward_cuda_kernel(
    scalar_t* __restrict__ alpha,
    scalar_t* __restrict__ weight,
    scalar_t* __restrict__ T,
    scalar_t* __restrict__ alphainv_last,
    int64_t* __restrict__ i_start,
    int64_t* __restrict__ i_end,
    const int n_rays,
    scalar_t* __restrict__ grad_weights,
    scalar_t* __restrict__ grad_last,
    scalar_t* __restrict__ grad) {

  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_ray<n_rays) {
    const int i_s = i_start[i_ray];
    const int i_e = i_end[i_ray];

    float back_cum = grad_last[i_ray] * alphainv_last[i_ray];
    for(int i=i_e-1; i>=i_s; --i) {
      grad[i] = grad_weights[i] * T[i] - back_cum / (1-alpha[i] + 1e-10);
      back_cum += grad_weights[i] * weight[i];
    }
  }
}

torch::Tensor alpha2weight_backward_cuda(
        torch::Tensor alpha, torch::Tensor weight, torch::Tensor T, torch::Tensor alphainv_last,
        torch::Tensor i_start, torch::Tensor i_end, const int n_rays,
        torch::Tensor grad_weights, torch::Tensor grad_last) {

  auto grad = torch::zeros_like(alpha);
  if(n_rays==0) {
    return grad;
  }

  const int threads = 256;
  const int blocks = (n_rays + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(alpha.type(), "alpha2weight_backward_cuda", ([&] {
    alpha2weight_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
        alpha.data<scalar_t>(),
        weight.data<scalar_t>(),
        T.data<scalar_t>(),
        alphainv_last.data<scalar_t>(),
        i_start.data<int64_t>(),
        i_end.data<int64_t>(),
        n_rays,
        grad_weights.data<scalar_t>(),
        grad_last.data<scalar_t>(),
        grad.data<scalar_t>());
  }));

  return grad;
}

