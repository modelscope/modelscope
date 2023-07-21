#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

/*
   helper function to skip oversampled points,
   especially near the foreground scene bbox boundary
   */
template <typename scalar_t>
__global__ void cumdist_thres_cuda_kernel(
        scalar_t* __restrict__ dist,
        const float thres,
        const int n_rays,
        const int n_pts,
        bool* __restrict__ mask) {
  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_ray<n_rays) {
    float cum_dist = 0;
    const int i_s = i_ray * n_pts;
    const int i_t = i_s + n_pts;
    int i;
    for(i=i_s; i<i_t; ++i) {
      cum_dist += dist[i];
      bool over = (cum_dist > thres);
      cum_dist *= float(!over);
      mask[i] = over;
    }
  }
}

torch::Tensor cumdist_thres_cuda(torch::Tensor dist, float thres) {
  const int n_rays = dist.size(0);
  const int n_pts = dist.size(1);
  const int threads = 256;
  const int blocks = (n_rays + threads - 1) / threads;
  auto mask = torch::zeros({n_rays, n_pts}, torch::dtype(torch::kBool).device(torch::kCUDA));
  AT_DISPATCH_FLOATING_TYPES(dist.type(), "cumdist_thres_cuda", ([&] {
    cumdist_thres_cuda_kernel<scalar_t><<<blocks, threads>>>(
        dist.data<scalar_t>(), thres,
        n_rays, n_pts,
        mask.data<bool>());
  }));
  return mask;
}
