# coding=utf-8
# Copyright 2020-present, AllenAI Authors, University of Illinois Urbana-Champaign,
# Intel Nervana Systems and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Binarizers take a (real value) matrix as input and produce a binary (values in {0,1}) mask of the same shape.
"""

import torch
from torch import autograd


class ThresholdBinarizer(autograd.Function):
    """
    Thresholdd binarizer.
    Computes a binary mask M from a real value matrix S such that `M_{i,j} = 1` if and only if `S_{i,j} > \tau`
    where `\tau` is a real value threshold.

    Implementation is inspired from:
        https://github.com/arunmallya/piggyback
        Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights
        Arun Mallya, Dillon Davis, Svetlana Lazebnik
    """

    @staticmethod
    def forward(ctx, inputs: torch.tensor, threshold: float, sigmoid: bool):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input matrix from which the binarizer computes the binary mask.
            threshold (`float`)
                The threshold value (in R).
            sigmoid (`bool`)
                If set to ``True``, we apply the sigmoid function to the `inputs` matrix before comparing to `threshold`.
                In this case, `threshold` should be a value between 0 and 1.
        Returns:
            mask (`torch.FloatTensor`)
                Binary matrix of the same size as `inputs` acting as a mask (1 - the associated weight is
                retained, 0 - the associated weight is pruned).
        """
        nb_elems = inputs.numel()
        nb_min = int(0.005 * nb_elems) + 1
        if sigmoid:
            mask = (torch.sigmoid(inputs) > threshold).type(inputs.type()).bool()
        else:
            mask = (inputs > threshold).type(inputs.type()).bool()
        if mask.sum() < nb_min:
            # We limit the pruning so that at least 0.5% (half a percent) of the weights are remaining
            k_threshold = inputs.flatten().kthvalue(max(nb_elems - nb_min, 1)).values
            mask = (inputs > k_threshold).type(inputs.type()).bool()
        return mask

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None, None


class TopKBinarizer(autograd.Function):
    """
    Top-k Binarizer.
    Computes a binary mask M from a real value matrix S such that `M_{i,j} = 1` if and only if `S_{i,j}`
    is among the k% highest values of S.

    Implementation is inspired from:
        https://github.com/allenai/hidden-networks
        What's hidden in a randomly weighted neural network?
        Vivek Ramanujan*, Mitchell Wortsman*, Aniruddha Kembhavi, Ali Farhadi, Mohammad Rastegari
    """

    @staticmethod
    def forward(ctx, inputs: torch.tensor, threshold: float, k_threshold=None):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input matrix from which the binarizer computes the binary mask.
            threshold (`float`)
                The percentage of weights to keep (the rest is pruned).
                `threshold` is a float between 0 and 1.
        Returns:
            mask (`torch.FloatTensor`)
                Binary matrix of the same size as `inputs` acting as a mask (1 - the associated weight is
                retained, 0 - the associated weight is pruned).
        """
        # Get the subnetwork by sorting the inputs and using the top threshold %
        if k_threshold is None:
            mask = inputs.clone()
            _, idx = inputs.flatten().sort(descending=True)
            j = int(threshold * inputs.numel())
            # flat_out and mask access the same memory.
            flat_out = mask.flatten()
            flat_out[idx[j:]] = 0
            flat_out[idx[:j]] = 1

            # if threshold == 1:
            #     k_threshold = -1000
            # else:
            #     n = inputs.numel()
            #     kth = min(max(n - (int(n * threshold) + 1), 1), n)
            #     k_threshold = inputs.flatten().kthvalue(kth).values
            # mask = (inputs > k_threshold).type(inputs.type())
        else:
            if threshold == 1.0:
                mask = (inputs > -1000).type(inputs.type())
            else:
                mask = (inputs > k_threshold).type(inputs.type())

        # # Get the subnetwork by get the kthvalue
        # # ==> This method will cause bug since if all the mask_scores are the same, the mask is all zero.
        # n = inputs.numel()
        # kth = max(n - (int(n * threshold) + 1), 1)
        # k_threshold = inputs.flatten().kthvalue(kth).values
        # mask = (inputs > k_threshold).type(inputs.type())

        # if torch.distributed.get_rank() == 0:
            # print("inputs:")
            # print(inputs, flush=True)
            # print('inputs isinf:')
            # print(torch.isinf(inputs), flush=True)
            # print('inputs isinf number:')
            # print(torch.isinf(inputs).sum(), flush=True)
            #
            # print('\n\n\nMask:')
            # print(mask, flush=True)
            # print('Mask isinf:')
            # print(torch.isinf(mask), flush=True)
            # print('Mask isinf number:')
            # print(torch.isinf(mask).sum(), flush=True)
            # print('Mask sum:')
            # print(torch.sum(mask), flush=True)
            #
            # print('inputs (mask_scores).mean(): ', inputs.mean().detach().cpu(), flush=True)
            # print('inputs (mask_scores).max(): ', inputs.max().detach().cpu(), flush=True)
            # print('inputs (mask_scores).min(): ', inputs.min().detach().cpu(), flush=True)
            # print('inputs is all 0?', (inputs != torch.tensor(0).type(inputs.type())).sum().detach().cpu().numpy())
            # print("\n\n\nMask ratio: {}/{}={}".format(mask.sum().detach().cpu(), inputs.numel(), (mask.sum().detach().cpu().numpy()) / float(inputs.numel())), flush=True)
            # print("threshold: {}".format(threshold), flush=True)

        return mask

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None, None


class MagnitudeBinarizer(object):
    """
    Magnitude Binarizer.
    Computes a binary mask M from a real value matrix S such that `M_{i,j} = 1` if and only if `S_{i,j}`
    is among the k% highest values of |S| (absolute value).

    Implementation is inspired from https://github.com/NervanaSystems/distiller/blob/2291fdcc2ea642a98d4e20629acb5a9e2e04b4e6/distiller/pruning/automated_gradual_pruner.py#L24
    """

    @staticmethod
    def apply(inputs: torch.tensor, threshold: float):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input matrix from which the binarizer computes the binary mask.
                This input marix is typically the weight matrix.
            threshold (`float`)
                The percentage of weights to keep (the rest is pruned).
                `threshold` is a float between 0 and 1.
        Returns:
            mask (`torch.FloatTensor`)
                Binary matrix of the same size as `inputs` acting as a mask (1 - the associated weight is
                retained, 0 - the associated weight is pruned).
        """
        # Get the subnetwork by sorting the inputs and using the top threshold %
        mask = inputs.clone()
        _, idx = inputs.abs().flatten().sort(descending=True)
        j = int(threshold * inputs.numel())
        # flat_out and mask access the same memory.
        flat_out = mask.flatten()
        flat_out[idx[j:]] = 0
        flat_out[idx[:j]] = 1
        # mask = mask.bool()
        return mask

        # # Get the subnetwork by sorting the inputs and using the top threshold
        # # ==> This method will cause bug since if all the mask_scores are the same, the mask is all zero.
        # n = inputs.numel()
        # kth = max(n - (int(n * threshold) + 1), 1)
        # k_threshold = inputs.abs().flatten().kthvalue(kth).values
        # mask = (inputs > k_threshold).type(inputs.type())
        # return mask

class MaskTaylor(autograd.Function):
    @staticmethod
    def forward(ctx, weight, mask):
        ctx.save_for_backward(weight, mask)
        return mask*weight

    @staticmethod
    def backward(ctx, gradOutput):
        weight, mask, = ctx.saved_tensors
        return gradOutput*mask, -torch.pow(gradOutput*weight, 2)
        # return gradOutput*mask, -torch.abs(gradOutput*weight)
