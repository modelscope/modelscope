'''
Copyright 2021 The Microsoft DeepSpeed Team
'''
# The file has been adapted from two fairscale files:
# (1) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/moe_layer.py
# (2) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/top2gate.py
# Git commit hash: 34df606902a240567a0d898037ece55c2f1336cf
# We retain the following license from the original files:

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron_util import mpu
from scipy.special import binom
from torch import Tensor, nn
from torch.nn import Module

from ..configuration import logger
from .mappings import drop_tokens, gather_tokens

try:
    # To enable Tutel MoE optimizations:
    # python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.1.x
    from tutel import moe as tutel_moe
    TUTEL_INSTALLED = True
except ImportError:
    # Fail silently so we don't spam logs unnecessarily if user isn't using tutel
    TUTEL_INSTALLED = False

try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    has_fused_layernorm = True

    class FusedLayerNorm(_FusedLayerNorm):

        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)
except ImportError:
    has_fused_layernorm = False

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

uniform_map: Dict[torch.device, Callable] = {}
gumbel_map: Dict[torch.device, Callable] = {}
exp_selection_uniform_map: Dict[torch.device, Callable] = {}


def multiplicative_jitter(x, device: torch.device, epsilon=1e-2):
    """
    Modified from switch transformer paper. mesh transformers
    Multiply values by a random number between 1-epsilon and 1+epsilon.
    Makes models more resilient to rounding errors introduced by bfloat16.
    This seems particularly important for logits.
    Args:
        x: a torch.tensor
        device: torch.device
        epsilon: a floating point value
    Returns:
        a jittered x.
    """
    if epsilon == 0:
        return x
    uniform = uniform_map.get(device)
    if uniform is None:
        uniform = torch.distributions.uniform.Uniform(
            low=torch.tensor(1.0 - epsilon, device=device),
            high=torch.tensor(1.0 + epsilon,
                              device=device)).rsample  # type: ignore
        uniform_map[device] = uniform
    return x * uniform(x.shape)


def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero,
                                                   one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup,
                input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))


# einsum rewrites are on par or more performant
# switch can be bubbled up in future
USE_EINSUM = True


# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.
def einsum(rule, a, b):
    if USE_EINSUM:
        return torch.einsum(rule, a, b)
    elif rule == 's,se->se':
        return a.reshape(a.shape[0], -1) * b
    elif rule == 'se,sc->sec':
        return a.unsqueeze(2) * b.unsqueeze(1)
    elif rule == 'se,se->s':
        return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).reshape(-1)
    elif rule == 'sec,sm->ecm':
        s = a.shape[0]
        e = a.shape[1]
        c = a.shape[2]
        m = b.shape[1]
        return torch.matmul(a.reshape(s, -1).t(), b).reshape(e, c, m)
    elif rule == 'sec,ecm->sm':
        return torch.matmul(
            a.reshape(a.shape[0], -1), b.reshape(-1, b.shape[-1]))
    elif rule == 'ks,ksm->sm':
        k = b.shape[0]
        s = b.shape[1]
        m = b.shape[2]
        # [k, s] -> [s, k] -> [s, 1, k]
        a = a.t().unsqueeze(1)
        # [k,s,m] -> [k, sm] -> [sm, k] -> [s, m, k]
        b = b.reshape(k, -1).t().reshape(s, m, k)
        # bmm([s, 1, k], [s, m, k]^t) -> [s, m, 1]
        return torch.bmm(a, b.transpose(1, 2)).squeeze(2)
    else:
        return torch.einsum(rule, a, b)


# The following functions are extracted and scripted
# because otherwise during a torch.jit.trace, the non-Tensor
# values used in the calculations get recorded as constants.
# torch.jit.script coerces them into Tensors and preserves
# their dynamic shapes. This enables ONNX export.
# We can't script the entire top1gating function because it
# includes stateful caching logic which is incompatible with ONNX.


@torch.jit.script
def _capacity(gates: Tensor, capacity_factor: Tensor,
              min_capacity: Tensor) -> Tensor:
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int64 when converting torch.topk but it doesn't.
    capacity = torch.ceil(
        (num_tokens / num_experts) * capacity_factor).to(torch.int64)
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)
    return capacity


@torch.jit.script
def _top_idx(source, k):
    return torch.topk(source, k=k, dim=0)[1]


@torch.jit.script
def _one_hot_to_float(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()


def top1gating(
        logits: Tensor,
        capacity_factor: float,
        min_capacity: int,
        used_token: Tensor = None,
        noisy_gate_policy: Optional[str] = None,
        drop_tokens: bool = True,
        use_rts: bool = True,
        use_tutel: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top1Gating on logits."""
    if noisy_gate_policy == 'RSample':
        logits_w_noise = logits + gumbel_rsample(
            logits.shape, device=logits.device)
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    capacity = _capacity(gates, torch.tensor(capacity_factor),
                         torch.tensor(min_capacity))

    # Create a mask for 1st's expert per token
    # noisy gating
    indices1_s = torch.argmax(
        logits_w_noise if noisy_gate_policy == 'RSample' else gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # mask only used tokens
    if used_token is not None:
        mask1 = einsum('s,se->se', used_token, mask1)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

    # if we don't want to drop any tokens
    if not drop_tokens:
        new_capacity = torch.max(exp_counts).to(logits.device)
        dist.all_reduce(
            new_capacity, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
        capacity = new_capacity

    # Compute l_aux
    alpha = torch.max(gates, dim=1).values.unsqueeze(1)
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.sum(me * ce) * num_experts

    # Random Token Selection
    if use_rts:
        uniform = exp_selection_uniform_map.get(logits.device)
        if uniform is None:
            uniform = torch.distributions.uniform.Uniform(
                low=torch.tensor(0.0, device=logits.device),
                high=torch.tensor(1.0, device=logits.device)).rsample
            exp_selection_uniform_map[logits.device] = uniform

        mask1_rand = mask1 * uniform(mask1.shape)
    else:
        mask1_rand = mask1

    assert logits.shape[0] >= min_capacity, \
        'No. of tokens (batch-size) should be greater than min_capacity. ' \
        'Either set min_capacity to 0 or increase your batch size.'

    top_idx = _top_idx(mask1_rand, capacity)

    new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)
    mask1 = new_mask1

    if use_tutel:
        # Tutel doesn't support index values masked with zero
        # so we need to replace masked indices with -1
        indices_mask = mask1.sum(dim=1) * num_experts - 1
        indices1_s = torch.min(indices1_s, indices_mask)

    # Compute locations in capacity buffer
    if use_tutel:
        locations1 = tutel_moe.fast_cumsum_sub_one(mask1)
    else:
        locations1 = torch.cumsum(mask1, dim=0) - 1

    if use_tutel:
        gates1_s = (gates * mask1).sum(dim=1)
        locations1_s = torch.sum(locations1 * mask1, dim=1)
        return l_aux, capacity, num_experts, [
            indices1_s,
        ], [
            locations1_s,
        ], [
            gates1_s,
        ], exp_counts, alpha

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    gates = gates * mask1_float

    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    combine_weights = einsum('se,sc->sec', gates, locations1_sc)

    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts, alpha


class TopKGate(Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(self,
                 model_dim: int,
                 num_experts: int,
                 k: int = 1,
                 capacity_factor: float = 1.0,
                 eval_capacity_factor: float = 1.0,
                 min_capacity: int = 8,
                 noisy_gate_policy: Optional[str] = None,
                 drop_tokens: bool = True,
                 use_rts: bool = True,
                 top_k_linear_strategy: str = 'standard') -> None:
        super().__init__()

        # Only top-1 are supported at the moment.
        if k != 1:
            raise ValueError('Only top-1 gatings are supported.')
        if top_k_linear_strategy == 'standard':
            self.wg = torch.nn.Linear(
                model_dim, num_experts, bias=False).float()
        elif top_k_linear_strategy == 'lsoftmax':
            self.wg = LSoftmaxLinearLayer(
                model_dim, num_experts, margin=1).float()
        else:
            raise ValueError(
                'Only standard or lsoftmax top-k-linear-strategy are supported.'
            )

        self.k = k
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.noisy_gate_policy = noisy_gate_policy
        self.wall_clock_breakdown = False
        self.gate_time = 0.0
        self.drop_tokens = drop_tokens
        self.use_rts = use_rts
        self.top_k_linear_strategy = top_k_linear_strategy

    def forward(
        self,
        input: torch.Tensor,
        used_token: torch.Tensor = None,
        use_tutel: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore

        if self.wall_clock_breakdown:
            self.timers('TopKGate').start()

        if self.top_k_linear_strategy == 'standard':
            if self.wg.weight.dtype != torch.float32:
                self.wg = self.wg.float()
        elif self.top_k_linear_strategy == 'lsoftmax':
            if self.wg.weight.weight.dtype != torch.float32:
                self.wg.weight = self.wg.weight.float()

        input_fp32 = input.float()
        # input jittering
        if self.noisy_gate_policy == 'Jitter' and self.training:
            input_fp32 = multiplicative_jitter(input_fp32, device=input.device)

        if self.k == 1:
            if self.top_k_linear_strategy == 'standard':
                logits = self.wg(input_fp32)
            elif self.top_k_linear_strategy == 'lsoftmax':
                logits = self.wg(input_fp32, input_fp32.device, self.training)

            gate_output = top1gating(
                logits, self.capacity_factor if self.training else
                self.eval_capacity_factor, self.min_capacity, used_token,
                self.noisy_gate_policy if self.training else None,
                self.drop_tokens, self.use_rts, use_tutel)

        if self.wall_clock_breakdown:
            self.timers('TopKGate').stop()
            self.gate_time = self.timers('TopKGate').elapsed(
                reset=False) * 1000

        return gate_output


class MOELayer(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """

    def __init__(self,
                 gate: Module,
                 experts: Module,
                 ep_group_name,
                 ep_size,
                 num_local_experts: int,
                 use_tutel: bool = False,
                 use_expert_residual_network: bool = False) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.ep_group = None
        self.ep_size = ep_size
        self.ep_group_name = ep_group_name
        self.num_local_experts = num_local_experts

        self.wall_clock_breakdown = False
        self.use_expert_residual_network = use_expert_residual_network

        if self.use_expert_residual_network:
            self.expert_network = nn.Sequential(*([
                ExpertResidualLayer(self.gate.model_dim) for _ in range(6)
            ]))  # noqa

        self.use_tutel = use_tutel and TUTEL_INSTALLED

        if self.use_tutel:
            logger.info('Using Tutel optimizations.')
        elif use_tutel and not TUTEL_INSTALLED:
            logger.info(
                'Tutel optimization requested but not installed Proceeding without Tutel.'
            )

    def _set_ep_group(self, ep_group):
        self.ep_group = ep_group

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:

        if self.wall_clock_breakdown:
            self.timers('moe').start()

        # Implement Algorithm 2 from GShard paper.
        d_model = input[0].shape[-1]
        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_input = input[0].reshape(-1, d_model)

        if self.use_tutel:
            self.l_aux, C, E, indices_, locations_, gates_, self.exp_counts, alpha = self.gate(
                reshaped_input, input[1], True)
            _, M = reshaped_input.size(0), reshaped_input.size(1)

            if not hasattr(self, '_tutel_dispatcher'):
                self._tutel_dispatcher = tutel_moe.fast_dispatcher(
                    E, C, M, dispatch_dtype=reshaped_input.dtype)
            self._tutel_dispatcher.update(
                indices_, locations_, gates_, capacity=C)
            dispatched_input = self._tutel_dispatcher.encode(reshaped_input)
        else:
            self.l_aux, combine_weights, dispatch_mask, self.exp_counts, alpha = self.gate(
                reshaped_input, input[1])
            dispatched_input = einsum('sec,sm->ecm',
                                      dispatch_mask.type_as(input[0]),
                                      reshaped_input)

        if self.wall_clock_breakdown:
            self.timers('falltoall').start()

        if mpu.get_expert_model_parallel_world_size() == 1:
            # If the non-expert is tensor-parallel, it will create
            # duplicate tokens on the tensor-parallel ranks.
            # Since our experts are not tensor-parallel, these duplicates
            # need to be dropped to ensure correctness.
            # this also doubles up as a communication optimization as we are
            # reducing the all-to-all communication volume.
            if self.use_tutel:
                # reshape tutel's output from [e*c,m] to [e,c,m]
                dispatched_input = dispatched_input.reshape(
                    self.ep_size * self.num_local_experts, -1, d_model)
            dispatched_input = drop_tokens(dispatched_input, dim=1)

        dispatched_input = _AllToAll.apply(self.ep_group, dispatched_input)

        if self.wall_clock_breakdown:
            self.timers('falltoall').stop()
            self.time_falltoall = self.timers('falltoall').elapsed(
                reset=False) * 1000

        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(self.ep_size,
                                                    self.num_local_experts, -1,
                                                    d_model)

        expert_output = self.experts(dispatched_input)

        if self.wall_clock_breakdown:
            self.timers('salltoall').start()

        expert_output = _AllToAll.apply(self.ep_group, expert_output)

        if self.wall_clock_breakdown:
            self.timers('salltoall').stop()
            self.time_salltoall = self.timers('salltoall').elapsed(
                reset=False) * 1000

        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(
            self.ep_size * self.num_local_experts, -1, d_model)

        if mpu.get_expert_model_parallel_world_size() == 1:
            # the dropped duplicate tokens need to be gathered on each
            # tensor parallel rank again for the tensor-parallel
            # non-expert of the next layer.
            expert_output = gather_tokens(expert_output, dim=1)

        if self.use_tutel:
            combined_output = self._tutel_dispatcher.decode(
                expert_output.view(E * C, M))
        else:
            combined_output = einsum('sec,ecm->sm',
                                     combine_weights.type_as(input[0]),
                                     expert_output)

        if self.use_expert_residual_network:
            combined_output = alpha * self.expert_network(combined_output) + (
                1 - alpha) * combined_output

        a = combined_output.reshape(input[0].shape)

        if self.wall_clock_breakdown:
            self.timers('moe').stop()
            self.time_moe = self.timers('moe').elapsed(reset=False) * 1000

        return a


class LSoftmaxLinearLayer(torch.nn.Module):

    def __init__(self, input_features, output_features, margin):
        super().__init__()
        self.input_dim = input_features  # number of input feature i.e. output of the last fc layer
        self.output_dim = output_features  # number of output = class numbers
        self.margin = margin  # m
        self.beta = 100
        self.beta_min = 0
        self.scale = 0.99
        self.num_experts = output_features
        # Initialize L-Softmax parameters
        self.weight = torch.nn.Linear(
            input_features, output_features, bias=False).float()
        self.divisor = math.pi / self.margin  # pi/m
        self.C_m_2n = torch.Tensor(binom(margin, range(0, margin + 1,
                                                       2)))  # C_m{2n}
        self.cos_powers = torch.Tensor(range(self.margin, -1, -2))  # m - 2n
        self.sin2_powers = torch.Tensor(range(len(self.cos_powers)))  # n
        self.signs = torch.ones(margin // 2 + 1)  # 1, -1, 1, -1, ...
        self.signs[1::2] = -1

    def calculate_cos_m_theta(self, cos_theta, device):
        sin2_theta = 1 - cos_theta**2
        cos_terms = cos_theta.unsqueeze(1)**self.cos_powers.to(
            device).unsqueeze(0)  # cos^{m - 2n}
        sin2_terms = (
            sin2_theta.unsqueeze(1)**self.sin2_powers.to(device).unsqueeze(0))

        cos_m_theta = (self.signs.to(device).unsqueeze(0)
                       * self.C_m_2n.to(device).unsqueeze(0) * cos_terms
                       * sin2_terms).sum(1)  # summation of all terms

        return cos_m_theta

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def find_k(self, cos):
        # to account for acos numerical errors
        eps = 1e-7
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

    def forward(self, input, device, training):
        if training:
            x, w = input, self.weight.float()
            beta = max(self.beta, self.beta_min)
            logit = w(x)
            indexes = range(logit.size(0))
            # target = torch.fmod(torch.randperm(logit.size(0)), self.num_experts)
            target = torch.fmod(
                torch.range(0,
                            logit.size(0) - 1), self.num_experts).long()
            logit_target = logit[indexes, target]

            # cos(theta) = w * x / ||w||*||x||
            w_target_norm = w.weight[:, target].norm(p=2, dim=0)

            x_norm = x.norm(p=2, dim=1)
            cos_theta_target = logit_target / (w_target_norm * x_norm + 1e-10)

            # equation 7
            cos_m_theta_target = self.calculate_cos_m_theta(
                cos_theta_target, device)

            # find k in equation 6
            k = self.find_k(cos_theta_target)

            # f_y_i
            logit_target_updated = w_target_norm * x_norm * ((
                (-1)**k * cos_m_theta_target) - 2 * k)
            logit_target_updated_beta = (logit_target_updated + beta
                                         * logit[indexes, target]) / (1 + beta)

            logit[indexes, target] = logit_target_updated_beta
            self.beta *= self.scale
            return logit
        else:
            return self.weight(input)


def LayerNorm(normalized_shape,
              eps=1e-5,
              elementwise_affine=True,
              export=False):
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        export = True
    if not export and torch.cuda.is_available() and has_fused_layernorm:
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class ExpertResidualLayer(torch.nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.norm = LayerNorm(embed_dim, export=False)
        self.ff1 = torch.nn.Linear(embed_dim, embed_dim * 4)
        self.ff2 = torch.nn.Linear(embed_dim * 4, embed_dim)
        self.ff2.weight.data.zero_()

    def forward(self, xs):
        return xs + self.ff2(torch.nn.functional.relu(self.ff1(self.norm(xs))))
