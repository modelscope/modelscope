#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void adam_upd_cuda_kernel(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ grad,
    scalar_t* __restrict__ exp_avg,
    scalar_t* __restrict__ exp_avg_sq,
    const size_t N,
    const float step_size, const float beta1, const float beta2, const float eps) {

  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index<N) {
    exp_avg[index] = beta1 * exp_avg[index] + (1-beta1) * grad[index];
    exp_avg_sq[index] = beta2 * exp_avg_sq[index] + (1-beta2) * grad[index] * grad[index];
    param[index] -= step_size * exp_avg[index] / (sqrt(exp_avg_sq[index]) + eps);
  }
}

template <typename scalar_t>
__global__ void masked_adam_upd_cuda_kernel(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ grad,
    scalar_t* __restrict__ exp_avg,
    scalar_t* __restrict__ exp_avg_sq,
    const size_t N,
    const float step_size, const float beta1, const float beta2, const float eps) {

  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index<N && grad[index]!=0) {
    exp_avg[index] = beta1 * exp_avg[index] + (1-beta1) * grad[index];
    exp_avg_sq[index] = beta2 * exp_avg_sq[index] + (1-beta2) * grad[index] * grad[index];
    param[index] -= step_size * exp_avg[index] / (sqrt(exp_avg_sq[index]) + eps);
  }
}

template <typename scalar_t>
__global__ void adam_upd_with_perlr_cuda_kernel(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ grad,
    scalar_t* __restrict__ exp_avg,
    scalar_t* __restrict__ exp_avg_sq,
    scalar_t* __restrict__ perlr,
    const size_t N,
    const float step_size, const float beta1, const float beta2, const float eps) {

  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index<N) {
    exp_avg[index] = beta1 * exp_avg[index] + (1-beta1) * grad[index];
    exp_avg_sq[index] = beta2 * exp_avg_sq[index] + (1-beta2) * grad[index] * grad[index];
    param[index] -= step_size * perlr[index] * exp_avg[index] / (sqrt(exp_avg_sq[index]) + eps);
  }
}

void adam_upd_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    const int step, const float beta1, const float beta2, const float lr, const float eps) {

  const size_t N = param.numel();

  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  const float step_size = lr * sqrt(1 - pow(beta2, (float)step)) / (1 - pow(beta1, (float)step));

  AT_DISPATCH_FLOATING_TYPES(param.type(), "adam_upd_cuda", ([&] {
    adam_upd_cuda_kernel<scalar_t><<<blocks, threads>>>(
        param.data<scalar_t>(),
        grad.data<scalar_t>(),
        exp_avg.data<scalar_t>(),
        exp_avg_sq.data<scalar_t>(),
        N, step_size, beta1, beta2, eps);
  }));
}

void masked_adam_upd_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    const int step, const float beta1, const float beta2, const float lr, const float eps) {

  const size_t N = param.numel();

  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  const float step_size = lr * sqrt(1 - pow(beta2, (float)step)) / (1 - pow(beta1, (float)step));

  AT_DISPATCH_FLOATING_TYPES(param.type(), "masked_adam_upd_cuda", ([&] {
    masked_adam_upd_cuda_kernel<scalar_t><<<blocks, threads>>>(
        param.data<scalar_t>(),
        grad.data<scalar_t>(),
        exp_avg.data<scalar_t>(),
        exp_avg_sq.data<scalar_t>(),
        N, step_size, beta1, beta2, eps);
  }));
}

void adam_upd_with_perlr_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor perlr,
    const int step, const float beta1, const float beta2, const float lr, const float eps) {

  const size_t N = param.numel();

  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  const float step_size = lr * sqrt(1 - pow(beta2, (float)step)) / (1 - pow(beta1, (float)step));

  AT_DISPATCH_FLOATING_TYPES(param.type(), "adam_upd_with_perlr_cuda", ([&] {
    adam_upd_with_perlr_cuda_kernel<scalar_t><<<blocks, threads>>>(
        param.data<scalar_t>(),
        grad.data<scalar_t>(),
        exp_avg.data<scalar_t>(),
        exp_avg_sq.data<scalar_t>(),
        perlr.data<scalar_t>(),
        N, step_size, beta1, beta2, eps);
  }));
}

