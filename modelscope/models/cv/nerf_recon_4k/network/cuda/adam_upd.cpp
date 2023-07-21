#include <torch/extension.h>

#include <vector>


void adam_upd_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    int step, float beta1, float beta2, float lr, float eps);

void masked_adam_upd_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    int step, float beta1, float beta2, float lr, float eps);

void adam_upd_with_perlr_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor perlr,
    int step, float beta1, float beta2, float lr, float eps);


// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void adam_upd(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    int step, float beta1, float beta2, float lr, float eps) {
  CHECK_INPUT(param);
  CHECK_INPUT(grad);
  CHECK_INPUT(exp_avg);
  CHECK_INPUT(exp_avg_sq);
  adam_upd_cuda(param, grad, exp_avg, exp_avg_sq,
          step, beta1, beta2, lr, eps);
}

void masked_adam_upd(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    int step, float beta1, float beta2, float lr, float eps) {
  CHECK_INPUT(param);
  CHECK_INPUT(grad);
  CHECK_INPUT(exp_avg);
  CHECK_INPUT(exp_avg_sq);
  masked_adam_upd_cuda(param, grad, exp_avg, exp_avg_sq,
          step, beta1, beta2, lr, eps);
}

void adam_upd_with_perlr(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor perlr,
    int step, float beta1, float beta2, float lr, float eps) {
  CHECK_INPUT(param);
  CHECK_INPUT(grad);
  CHECK_INPUT(exp_avg);
  CHECK_INPUT(exp_avg_sq);
  adam_upd_with_perlr_cuda(param, grad, exp_avg, exp_avg_sq, perlr,
          step, beta1, beta2, lr, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("adam_upd", &adam_upd,
          "Adam update");
  m.def("masked_adam_upd", &masked_adam_upd,
          "Adam update ignoring zero grad");
  m.def("adam_upd_with_perlr", &adam_upd_with_perlr,
          "Adam update ignoring zero grad with per-voxel lr");
}
