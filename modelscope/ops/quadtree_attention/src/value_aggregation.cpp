// Copyright (c) Alibaba, Inc. and its affiliates.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "value_aggregation.h"
//extern THCState *state;
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void value_aggregation_cuda_forward(
                    at::Tensor score, // B, N, K, H
                    at::Tensor value, // B, M, H, D
                    at::Tensor index, // B, N, K, H
                    at::Tensor output)// B, N, H, D
{
    CHECK_INPUT(score);
    CHECK_INPUT(value);
    CHECK_INPUT(index);
    auto score_size = score.sizes();
    auto value_size = value.sizes();
    int B = score_size[0];
    int N = score_size[1];
    int K = score_size[2];
    int H = score_size[3];
    int M = value_size[1];
    int D = value_size[3];


    value_aggregation_forward_kernel(score.data<float>(), value.data<float>(),
        index.data<long>(), output.data<float>(), B, N, K, H, M, D,
        at::cuda::getCurrentCUDAStream());
}

void value_aggregation_cuda_backward(
                    at::Tensor grad_output, // B, N, H, D
                    at::Tensor score, // B, N, K, H
                    at::Tensor value, // B, M, H, D
                    at::Tensor index, // B, N, K, H
                    at::Tensor grad_score, // B, N, K, H
                    at::Tensor grad_value // B, M, H, D
                    )
{
    CHECK_INPUT(score);
    CHECK_INPUT(value);
    CHECK_INPUT(index);
    CHECK_INPUT(grad_output);

    auto score_size = score.sizes();
    auto value_size = value.sizes();
    int B = score_size[0];
    int N = score_size[1];
    int K = score_size[2];
    int H = score_size[3];
    int M = value_size[1];
    int D = value_size[3];


    value_aggregation_backward_kernel(grad_output.data<float>(), score.data<float>(),
        value.data<float>(), index.data<long>(), grad_score.data<float>(), grad_value.data<float>(),
        B, N, K, H, M, D, at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("value_aggregation_forward", &value_aggregation_cuda_forward, "value forward (CUDA)");
  m.def("value_aggregation_backward", &value_aggregation_cuda_backward, "value backward (CUDA)");
}
