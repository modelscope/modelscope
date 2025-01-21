'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

import typing

import torch
from megatron_util import mpu

from .experts import Experts
from .sharded_moe import MOELayer, TopKGate


class MoE(torch.nn.Module):

    def __init__(self,
                 hidden_size,
                 expert,
                 num_experts=1,
                 ep_size=1,
                 k=1,
                 capacity_factor=1.,
                 eval_capacity_factor=1.,
                 min_capacity=4,
                 use_residual=False,
                 noisy_gate_policy: typing.Optional[str] = None,
                 drop_tokens: bool = True,
                 use_rts=True,
                 use_tutel: bool = False,
                 top_k_linear_strategy: str = 'normal',
                 use_expert_residual_network: bool = False):
        super(MoE, self).__init__()
        self.use_residual = use_residual
        assert num_experts % ep_size == 0, f'Number of experts ({num_experts}) should ' \
                                           f'be divisible by expert parallel size ({ep_size})'
        self.ep_size = ep_size
        self.expert_group_name = f'ep_size_{self.ep_size}'
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.ep_size

        assert noisy_gate_policy is None or noisy_gate_policy in ['None', 'Jitter', 'RSample'], \
            'Unsupported noisy_gate_policy: ' + noisy_gate_policy

        experts = Experts(expert, self.num_local_experts,
                          self.expert_group_name)
        self.deepspeed_moe = MOELayer(
            TopKGate(
                hidden_size,
                num_experts,
                k,
                capacity_factor,
                eval_capacity_factor,
                min_capacity,
                noisy_gate_policy,
                drop_tokens,
                use_rts,
                top_k_linear_strategy=top_k_linear_strategy),
            experts,
            self.expert_group_name,
            self.ep_size,
            self.num_local_experts,
            use_tutel=use_tutel,
            use_expert_residual_network=use_expert_residual_network)

        self.deepspeed_moe._set_ep_group(
            mpu.get_expert_parallel_group(self.expert_group_name))

        if self.use_residual:
            self.mlp = expert
            # coefficient is used for weighted sum of the output of expert and mlp
            self.coefficient = torch.nn.Linear(hidden_size, 2)

    def forward(self, hidden_states, used_token=None):
        """ MoE forward

        Arguments:
            hidden_states (Tensor): input to the layer
            used_token (Tensor, optional): default: None, mask only used tokens

        Returns:
            A tuple including output, gate loss, and expert count.

            * output (Tensor): output of the model

            * l_aux (Tensor): gate loss value

            * exp_counts (int): expert count
        """
        output = self.deepspeed_moe(hidden_states, used_token)
        if self.use_residual:
            # Residual MoE
            output_mlp = self.mlp(hidden_states)
            if type(output_mlp) is tuple:
                output_mlp = output_mlp[0]  # Ignore the bias term for now
            coef = self.coefficient(hidden_states)
            coef = torch.nn.functional.softmax(coef, dim=1)
            output = output * coef[..., 0:1] + output_mlp * coef[..., 1:]
        return output, self.deepspeed_moe.l_aux, self.deepspeed_moe.exp_counts
