# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn as nn

from modelscope.utils.torch_utils import is_master


class SparseBinarizer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mask_scores, sparsity):
        num_prune = int(mask_scores.numel() * sparsity)
        prune_indices = torch.argsort(mask_scores.reshape(-1))[:num_prune]
        mask = mask_scores.clone().fill_(1)
        mask.reshape(-1)[prune_indices] = 0.0
        return mask

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None


class SparseLinear(nn.Module):
    """
    Fully Connected layer with on the fly adaptive mask.
    """

    def __init__(
        self,
        module,
        pruning_method='pst',
        weight_rank=8,
        weight_beta=1.0,
        mask_rank=8,
        mask_alpha1=1.0,
        mask_alpha2=1.0,
    ):
        super(SparseLinear, self).__init__()
        self.module = module
        out_features = self.module.weight.shape[0]
        in_features = self.module.weight.shape[1]

        self.weight = self.module.weight
        self.module.weight = None
        self.module._parameters.pop('weight')

        self.pruning_method = pruning_method

        self.cur_sparsity = 0.0

        if self.pruning_method == 'pst':
            self.weight_rank = weight_rank
            self.weight_beta = weight_beta
            self.mask_rank = mask_rank
            self.mask_alpha1 = mask_alpha1
            self.mask_alpha2 = mask_alpha2

            # create trainable params
            self.weight_U = nn.Parameter(
                torch.randn(out_features, self.weight_rank).to(
                    device=self.weight.device, dtype=self.weight.dtype))
            self.weight_V = nn.Parameter(
                torch.zeros(self.weight_rank, in_features).to(
                    device=self.weight.device, dtype=self.weight.dtype))

            self.mask_scores_A = nn.Parameter(
                torch.randn(out_features, self.mask_rank).to(
                    device=self.weight.device, dtype=self.weight.dtype))
            self.mask_scores_B = nn.Parameter(
                torch.zeros(self.mask_rank, in_features).to(
                    device=self.weight.device, dtype=self.weight.dtype))
            self.mask_scores_R = nn.Parameter(
                torch.zeros(out_features).to(
                    device=self.weight.device, dtype=self.weight.dtype))
            self.mask_scores_C = nn.Parameter(
                torch.zeros(in_features).to(
                    device=self.weight.device, dtype=self.weight.dtype))

            self.weight.requires_grad = False
            if self.module.bias is not None:
                self.module.bias.requires_grad = False

    def forward(self, *inputs):
        if self.pruning_method == 'pst':
            weight = self.weight + self.weight_beta * self.weight_U @ self.weight_V
            mask_scores = (
                weight.abs()
                + self.mask_alpha1 * self.mask_scores_A @ self.mask_scores_B
                + self.mask_alpha2 * (self.mask_scores_R.unsqueeze(1)
                                      + self.mask_scores_C.unsqueeze(0)))

            mask = SparseBinarizer.apply(mask_scores, self.cur_sparsity)
            masked_weight = mask * weight

            self.module.weight = masked_weight
            return self.module(*inputs)
        else:
            return self.module(*inputs)

    def convert(self):
        if self.pruning_method == 'pst':
            weight = self.weight + self.weight_beta * self.weight_U @ self.weight_V
            mask_scores = (
                weight.abs()
                + self.mask_alpha1 * self.mask_scores_A @ self.mask_scores_B
                + self.mask_alpha2 * (self.mask_scores_R.unsqueeze(1)
                                      + self.mask_scores_C.unsqueeze(0)))

            mask = SparseBinarizer.apply(mask_scores, self.cur_sparsity)

            masked_weight = mask * weight
            self.module.weight = nn.Parameter(masked_weight.data)


def _setattr(model, name, module):
    name_list = name.split('.')
    for name in name_list[:-1]:
        model = getattr(model, name)
    setattr(model, name_list[-1], module)


def convert_sparse_network(
    model,
    pruning_method,
    weight_rank,
    weight_beta,
    mask_rank,
    mask_alpha1,
    mask_alpha2,
    logger=None,
):
    compress_module = [nn.Linear]
    try:
        from megatron_util import mpu
        compress_module.extend(
            [mpu.RowParallelLinear, mpu.ColumnParallelLinear])
    except ImportError:
        pass

    for name, module in model.named_modules():
        if type(module) in compress_module:
            new_module = SparseLinear(
                module,
                pruning_method,
                weight_rank,
                weight_beta,
                mask_rank,
                mask_alpha1,
                mask_alpha2,
            )

            # replace original module by new sparse module
            _setattr(model, name, new_module)

            if is_master():
                if logger:
                    logger.info(f'convert {name} to sparse module.')
                else:
                    print(f'convert {name} to sparse module.')


def update_network_sparsity(model, sparsity):
    for name, module in model.named_modules():
        if isinstance(module, SparseLinear):
            module.cur_sparsity = sparsity


def schedule_sparsity_ratio(
    step,
    total_step,
    frequency,
    initial_warmup,
    final_warmup,
    initial_sparsity,
    final_sparsity,
):
    if step <= initial_warmup * total_step:
        sparsity = initial_sparsity
    elif step > (total_step - final_warmup * total_step):
        sparsity = final_sparsity
    else:
        spars_warmup_steps = initial_warmup * total_step
        spars_schedu_steps = (final_warmup + initial_warmup) * total_step
        step = (step - spars_warmup_steps) // frequency * frequency
        mul_coeff = 1 - step / (total_step - spars_schedu_steps)
        sparsity = final_sparsity + (initial_sparsity - final_sparsity) * (
            mul_coeff**3)
    return sparsity


def generate_sparse_model(model, logger=None):
    # generate sparse weight for saving
    for name, module in model.named_modules():
        if isinstance(module, SparseLinear):
            module.convert()

            _setattr(model, name, module.module)

            if is_master():
                if logger:
                    logger.info(f'convert {name} weight to sparse weight, \
                            sparsity ratio={torch.mean(1.0*(module.module.weight==0)).item()}.'
                                )
                else:
                    print(f'convert {name} weight to sparse, \
                            sparsity ratio={torch.mean(1.0*(module.module.weight==0)).item()}.'
                          )
