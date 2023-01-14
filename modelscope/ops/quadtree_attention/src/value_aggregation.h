// Copyright (c) Alibaba, Inc. and its affiliates.

#ifndef _VALUE_AGGREGATION_
#define _VALUE_AGGREGATION_
#include <torch/extension.h>
#include <vector>

void value_aggregation_forward_kernel(float* score, // B, N, K, H
                    float* value, // B, M, H, D
                    long* index, // B, N, K, H
                    float* output, // B, N, H, D
                    int B, int N, int K, int H, int M, int D, cudaStream_t stream
                    );

void value_aggregation_cuda_forward(at::Tensor score, at::Tensor value, at::Tensor index, at::Tensor output);

void value_aggregation_backward_kernel(float* grad_output, float* score, float* value,long* index, float* grad_score, float* grad_value, int B, int N, int K, int H, int M, int D, cudaStream_t stream);

#endif // _VALUE_AGGREGATION_
