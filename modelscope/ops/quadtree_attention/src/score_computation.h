// Copyright (c) Alibaba, Inc. and its affiliates.

#ifndef _Score_CUDA
#define _Score_CUDA
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> score_cuda_forward(torch::Tensor input1,       //query t: N, H, W, C1
                                             torch::Tensor input2,       //scene : N, H, W, C1
                                            torch::Tensor index);       //scene : N, H, W, C1



std::vector<at::Tensor> ScoreData_ongpu(at::Tensor input1,       //query t: N, H, W, C1
                                             at::Tensor input2,       //scene : N, H, W, C1
                                            at::Tensor index);       //scene : N, H, W, C1


std::vector<torch::Tensor> ScoreData_backward_ongpu(torch::Tensor grad_output1, //B,N,C,group_num
                      torch::Tensor input1, //scene : N, H, W, C1
                      torch::Tensor input2, // scene coords: N, H, W, 3
                      torch::Tensor index); //tensor: B, N, K

#endif
