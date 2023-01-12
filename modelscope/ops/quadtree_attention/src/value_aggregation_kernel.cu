// Copyright (c) Alibaba, Inc. and its affiliates.

#include <vector>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <vector>
#include "value_aggregation.h"
#include "THC/THCAtomics.cuh"
#include <stdio.h>
#include "utils.h"

#define ROUND_OFF 50000

#define CUDA_NUM_THREADS 1024
#define WARPS_PER_BLOCK 1
#define THREADS_PER_WARP 32

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define GET_BLOCKS(n, t) (n+t-1) / t

__global__ void ValueAggregationForwardFunc(float* score, float* value, long* index, float* output, int B, int N, int K, int H, int M, int D) {
  ///*
  long LENGTH = B*N*H*D;
  CUDA_KERNEL_LOOP(cur_idx, LENGTH){
      long d_idx = cur_idx % D;
      long h_idx = (cur_idx - d_idx) / D % H;
      long n_idx = (cur_idx - d_idx - h_idx * D) / D / H % N;
      long b_idx = (cur_idx - d_idx - h_idx * D - n_idx * H * D) / D / H / N;
      if (cur_idx < LENGTH) {
        long score_start_idx = b_idx * N * K * H + n_idx * K * H + h_idx;
        long value_start_idx = b_idx * M * H * D + h_idx * D + d_idx;

        float out_val = 0;
        for(int k_idx = 0; k_idx < K; k_idx++){
            int score_idx = score_start_idx + k_idx * H;
            int value_idx = value_start_idx + index[score_idx] * H * D;
            out_val += score[score_idx] * value[value_idx];
        }
        output[cur_idx] = out_val;
      }
  }
}


void value_aggregation_forward_kernel(float* score, float* value, long* index, float* ouput, int B, int N, int K, int H, int M, int D, cudaStream_t stream){
  ValueAggregationForwardFunc
    <<<GET_BLOCKS(B*N*H*D, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, stream>>>(score, value, index, ouput, B, N, K, H, M, D);

  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
    throw std::runtime_error(Formatter()
                             << "CUDA kernel failed : " << std::to_string(err));
}

__global__ void ValueAggregationBackwardFunc(float* grad_output, float* score, float* value, long* index, float* grad_score,
         float* grad_value, int B, int N, int K, int H, int M, int D) {
  long LENGTH = B*N*K*H;
  CUDA_KERNEL_LOOP(cur_idx, LENGTH){
      long h_idx = cur_idx % H;
      long k_idx = (cur_idx - h_idx) / H % K;
      long n_idx = (cur_idx - h_idx - k_idx * H) / H / K % N;
      long b_idx = (cur_idx - h_idx - k_idx * H - n_idx * H * K) / H / K / N;

      if (cur_idx < LENGTH) {
        long output_start_idx = b_idx * N * H * D + n_idx * H * D + h_idx * D;
        long value_start_idx = b_idx * M * H * D + h_idx * D;
        for (int d_idx = 0; d_idx < D; d_idx ++){
            long output_idx = output_start_idx + d_idx;
            long value_idx = value_start_idx + index[cur_idx] * H * D + d_idx;
            auto grad_output_val = grad_output[output_idx];
            grad_score[cur_idx] += grad_output_val * value[value_idx];
            gpuAtomicAdd(&grad_value[value_idx], grad_output_val * score[cur_idx]);
        }
      }
  }
}

void value_aggregation_backward_kernel(float* grad_output, float* score, float* value, long* index, float* grad_score, float* grad_value, int B, int N, int K, int H, int M, int D, cudaStream_t stream){
  ValueAggregationBackwardFunc
    <<<GET_BLOCKS(B*N*K*H, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, stream>>>(grad_output, score, value, index, grad_score, grad_value, B, N, K, H, M, D);

  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
    throw std::runtime_error(Formatter()
                             << "CUDA kernel failed : " << std::to_string(err));
}
