#include <torch/extension.h>

#include <vector>


std::vector<torch::Tensor> infer_t_minmax_cuda(
        torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor xyz_min, torch::Tensor xyz_max,
        const float near, const float far);

torch::Tensor infer_n_samples_cuda(torch::Tensor rays_d, torch::Tensor t_min, torch::Tensor t_max, const float stepdist);

std::vector<torch::Tensor> infer_ray_start_dir_cuda(torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor t_min);

std::vector<torch::Tensor> sample_pts_on_rays_cuda(
        torch::Tensor rays_o, torch::Tensor rays_d,
        torch::Tensor xyz_min, torch::Tensor xyz_max,
        const float near, const float far, const float stepdist);

std::vector<torch::Tensor> sample_ndc_pts_on_rays_cuda(
        torch::Tensor rays_o, torch::Tensor rays_d,
        torch::Tensor xyz_min, torch::Tensor xyz_max,
        const int N_samples);

torch::Tensor sample_bg_pts_on_rays_cuda(
        torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor t_max,
        const float bg_preserve, const int N_samples);

torch::Tensor maskcache_lookup_cuda(torch::Tensor world, torch::Tensor xyz, torch::Tensor xyz2ijk_scale, torch::Tensor xyz2ijk_shift);

std::vector<torch::Tensor> raw2alpha_cuda(torch::Tensor density, const float shift, const float interval);
std::vector<torch::Tensor> raw2alpha_nonuni_cuda(torch::Tensor density, const float shift, torch::Tensor interval);

torch::Tensor raw2alpha_backward_cuda(torch::Tensor exp, torch::Tensor grad_back, const float interval);
torch::Tensor raw2alpha_nonuni_backward_cuda(torch::Tensor exp, torch::Tensor grad_back, torch::Tensor interval);

std::vector<torch::Tensor> alpha2weight_cuda(torch::Tensor alpha, torch::Tensor ray_id, const int n_rays);

torch::Tensor alpha2weight_backward_cuda(
        torch::Tensor alpha, torch::Tensor weight, torch::Tensor T, torch::Tensor alphainv_last,
        torch::Tensor i_start, torch::Tensor i_end, const int n_rays,
        torch::Tensor grad_weights, torch::Tensor grad_last);


#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> infer_t_minmax(
        torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor xyz_min, torch::Tensor xyz_max,
        const float near, const float far) {
  CHECK_INPUT(rays_o);
  CHECK_INPUT(rays_d);
  CHECK_INPUT(xyz_min);
  CHECK_INPUT(xyz_max);
  return infer_t_minmax_cuda(rays_o, rays_d, xyz_min, xyz_max, near, far);
}

torch::Tensor infer_n_samples(torch::Tensor rays_d, torch::Tensor t_min, torch::Tensor t_max, const float stepdist) {
  CHECK_INPUT(rays_d);
  CHECK_INPUT(t_min);
  CHECK_INPUT(t_max);
  return infer_n_samples_cuda(rays_d, t_min, t_max, stepdist);
}

std::vector<torch::Tensor> infer_ray_start_dir(torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor t_min) {
  CHECK_INPUT(rays_o);
  CHECK_INPUT(rays_d);
  CHECK_INPUT(t_min);
  return infer_ray_start_dir_cuda(rays_o, rays_d, t_min);
}

std::vector<torch::Tensor> sample_pts_on_rays(
        torch::Tensor rays_o, torch::Tensor rays_d,
        torch::Tensor xyz_min, torch::Tensor xyz_max,
        const float near, const float far, const float stepdist) {
  CHECK_INPUT(rays_o);
  CHECK_INPUT(rays_d);
  CHECK_INPUT(xyz_min);
  CHECK_INPUT(xyz_max);
  assert(rays_o.dim()==2);
  assert(rays_o.size(1)==3);
  return sample_pts_on_rays_cuda(rays_o, rays_d, xyz_min, xyz_max, near, far, stepdist);
}

std::vector<torch::Tensor> sample_ndc_pts_on_rays(
        torch::Tensor rays_o, torch::Tensor rays_d,
        torch::Tensor xyz_min, torch::Tensor xyz_max,
        const int N_samples) {
  CHECK_INPUT(rays_o);
  CHECK_INPUT(rays_d);
  CHECK_INPUT(xyz_min);
  CHECK_INPUT(xyz_max);
  assert(rays_o.dim()==2);
  assert(rays_o.size(1)==3);
  return sample_ndc_pts_on_rays_cuda(rays_o, rays_d, xyz_min, xyz_max, N_samples);
}

torch::Tensor sample_bg_pts_on_rays(
        torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor t_max,
        const float bg_preserve, const int N_samples) {
  CHECK_INPUT(rays_o);
  CHECK_INPUT(rays_d);
  CHECK_INPUT(t_max);
  return sample_bg_pts_on_rays_cuda(rays_o, rays_d, t_max, bg_preserve, N_samples);
}

torch::Tensor maskcache_lookup(torch::Tensor world, torch::Tensor xyz, torch::Tensor xyz2ijk_scale, torch::Tensor xyz2ijk_shift) {
  CHECK_INPUT(world);
  CHECK_INPUT(xyz);
  CHECK_INPUT(xyz2ijk_scale);
  CHECK_INPUT(xyz2ijk_shift);
  assert(world.dim()==3);
  assert(xyz.dim()==2);
  assert(xyz.size(1)==3);
  return maskcache_lookup_cuda(world, xyz, xyz2ijk_scale, xyz2ijk_shift);
}

std::vector<torch::Tensor> raw2alpha(torch::Tensor density, const float shift, const float interval) {
  CHECK_INPUT(density);
  assert(density.dim()==1);
  return raw2alpha_cuda(density, shift, interval);
}
std::vector<torch::Tensor> raw2alpha_nonuni(torch::Tensor density, const float shift, torch::Tensor interval) {
  CHECK_INPUT(density);
  assert(density.dim()==1);
  return raw2alpha_nonuni_cuda(density, shift, interval);
}

torch::Tensor raw2alpha_backward(torch::Tensor exp, torch::Tensor grad_back, const float interval) {
  CHECK_INPUT(exp);
  CHECK_INPUT(grad_back);
  return raw2alpha_backward_cuda(exp, grad_back, interval);
}
torch::Tensor raw2alpha_nonuni_backward(torch::Tensor exp, torch::Tensor grad_back, torch::Tensor interval) {
  CHECK_INPUT(exp);
  CHECK_INPUT(grad_back);
  return raw2alpha_nonuni_backward_cuda(exp, grad_back, interval);
}

std::vector<torch::Tensor> alpha2weight(torch::Tensor alpha, torch::Tensor ray_id, const int n_rays) {
  CHECK_INPUT(alpha);
  CHECK_INPUT(ray_id);
  assert(alpha.dim()==1);
  assert(ray_id.dim()==1);
  assert(alpha.sizes()==ray_id.sizes());
  return alpha2weight_cuda(alpha, ray_id, n_rays);
}

torch::Tensor alpha2weight_backward(
        torch::Tensor alpha, torch::Tensor weight, torch::Tensor T, torch::Tensor alphainv_last,
        torch::Tensor i_start, torch::Tensor i_end, const int n_rays,
        torch::Tensor grad_weights, torch::Tensor grad_last) {
  CHECK_INPUT(alpha);
  CHECK_INPUT(weight);
  CHECK_INPUT(T);
  CHECK_INPUT(alphainv_last);
  CHECK_INPUT(i_start);
  CHECK_INPUT(i_end);
  CHECK_INPUT(grad_weights);
  CHECK_INPUT(grad_last);
  return alpha2weight_backward_cuda(
          alpha, weight, T, alphainv_last,
          i_start, i_end, n_rays,
          grad_weights, grad_last);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("infer_t_minmax", &infer_t_minmax, "Inference t_min and t_max of ray-bbox intersection");
  m.def("infer_n_samples", &infer_n_samples, "Inference the number of points to sample on each ray");
  m.def("infer_ray_start_dir", &infer_ray_start_dir, "Inference the starting point and shooting direction of each ray");
  m.def("sample_pts_on_rays", &sample_pts_on_rays, "Sample points on rays");
  m.def("sample_ndc_pts_on_rays", &sample_ndc_pts_on_rays, "Sample points on rays");
  m.def("sample_bg_pts_on_rays", &sample_bg_pts_on_rays, "Sample points on bg");
  m.def("maskcache_lookup", &maskcache_lookup, "Lookup to skip know freespace.");
  m.def("raw2alpha", &raw2alpha, "Raw values [-inf, inf] to alpha [0, 1].");
  m.def("raw2alpha_backward", &raw2alpha_backward, "Backward pass of the raw to alpha");
  m.def("raw2alpha_nonuni", &raw2alpha_nonuni, "Raw values [-inf, inf] to alpha [0, 1].");
  m.def("raw2alpha_nonuni_backward", &raw2alpha_nonuni_backward, "Backward pass of the raw to alpha");
  m.def("alpha2weight", &alpha2weight, "Per-point alpha to accumulated blending weight");
  m.def("alpha2weight_backward", &alpha2weight_backward, "Backward pass of alpha2weight");
}
