import os
import re
import types
from dataclasses import dataclass, field
from typing import Union

import torch
from torch import nn

from modelscope import snapshot_download
from modelscope.utils.constant import ModelFile
from .base import SwiftConfig


@dataclass
class PromptConfig(SwiftConfig):
    """
    The configuration class for the prompt module.

    Visual prompt tuning (VPT) is proposed to initialize tunable prompt tokens
    and prepend to the original tokens in the first layer or multiple layers.
    'Visual Prompt Tuning' by Jia et al.(2022)
    See https://arxiv.org/abs/2203.12119

    Here we apply the VPT to other fields.

    Args:
        dim: The dimension of the hidden states
        module_layer_name: The layer module to be replaced, in regex format
        embedding_pos: The position of the embedding tensor
        attention_mask_pos: The position of the attention mask
        attention_mask_value: The value to pad to the attention mask
        prompt_length: The length of the prompt tokens
        only_prompt_trainable: Whether to train only prompt
        attach_front: When set to True, prompt is attached in front of the embedding
        extract_embedding: Whether the embedding is extracted at final stage to keep the same dims with inputs
        pretrained_weights: The pretrained prompt weights. Can be a local dir, local file,
            or a model id from modelscope
    """

    dim: int = field(metadata={'help': 'The dimension of the hidden states'})

    module_layer_name: str = field(
        metadata={'help': 'The layer module to be replaced, in regex format'})

    embedding_pos: Union[str, int] = field(
        metadata={'help': 'The position of the embedding tensor'})

    attention_mask_pos: Union[str, int] = field(
        default=None, metadata={'help': 'The position of the attention mask'})

    attention_mask_value: Union[float, int, bool] = field(
        default=0.,
        metadata={'help': 'The value to pad to the attention mask'})

    prompt_length: int = field(
        default=16, metadata={'help': 'The length of the prompt tokens'})

    only_prompt_trainable: bool = field(
        default=True, metadata={'help': 'Whether to train only prompt'})

    attach_front: bool = field(
        default=True,
        metadata={
            'help':
            'When set to True, prompt is attached in front of the embedding'
        })

    extract_embedding: bool = field(
        default=False,
        metadata={
            'help':
            'Whether the embedding is extracted at final stage to keep the same dims with inputs'
        })

    pretrained_weights: str = field(
        default=None,
        metadata={
            'help':
            'The pretrained prompt weights. Can be a local dir, local file, or a model id from modelscope'
        })


class Prompt:

    @staticmethod
    def prepare_model(model: nn.Module, config: PromptConfig):
        module_keys = [key for key, _ in model.named_modules()]
        match_module_keys = []
        for module_key in module_keys:
            if re.fullmatch(config.module_layer_name, module_key):  # noqa
                module = model.get_submodule(module_key)

                def _forward(self, *args, **kwargs):
                    if isinstance(config.embedding_pos, int):
                        input_embedding = args[config.embedding_pos]
                    else:
                        input_embedding = kwargs[config.embedding_pos]

                    input_embedding = getattr(
                        self, 'prompt').forward(input_embedding)
                    if isinstance(config.embedding_pos, int):
                        args = type(args)(
                            args[0:config.embedding_pos] + (input_embedding, )
                            + args[config.embedding_pos + 1:])
                    else:
                        kwargs[config.embedding_pos] = input_embedding

                    if config.attention_mask_pos:
                        attention_mask = None
                        if isinstance(config.attention_mask_pos, int):
                            attention_mask = args[config.attention_mask_pos]
                        elif isinstance(config.attention_mask_pos, str):
                            attention_mask = kwargs[config.attention_mask_pos]

                        if attention_mask is not None:
                            attention_mask = getattr(
                                self,
                                'prompt').patch_attention_mask(attention_mask)
                        if isinstance(config.attention_mask_pos, int):
                            args = type(args)(
                                args[0:config.attention_mask_pos]
                                + (attention_mask, )
                                + args[config.attention_mask_pos + 1:])
                        else:
                            kwargs[config.attention_mask_pos] = attention_mask

                    forward_output = self.forward_origin(*args, **kwargs)
                    if config.extract_embedding:
                        forward_output = getattr(
                            self, 'prompt').extract(forward_output)

                    return forward_output

                module.forward_origin = module.forward
                module.forward = types.MethodType(_forward, module)

                if isinstance(config.dim, list):
                    input_dim = config.dim[len(match_module_keys)]
                else:
                    input_dim = config.dim

                prompt_module = PromptModule(input_dim,
                                             int(module_key.rsplit('.')[-1]),
                                             config.prompt_length,
                                             config.attention_mask_value,
                                             config.attach_front)
                setattr(module, 'prompt', prompt_module)
                match_module_keys.append(module_key)

        if config.only_prompt_trainable:
            for n, p in model.named_parameters():
                if 'prompt' not in n:
                    p.requires_grad = False

        def state_dict_hook(module, destination, prefix, local_metadata):
            return {
                key: value
                for key, value in destination.items() if 'prompt' in key
            }

        model.state_dict_hook_handle = model._register_state_dict_hook(
            state_dict_hook)

        def load_state_dict(self, state_dict, strict=True):
            return self.load_state_dict_origin(state_dict, False)

        model.load_state_dict_origin = model.load_state_dict
        model.load_state_dict = types.MethodType(load_state_dict, model)

        if config.pretrained_weights is not None:
            if not os.path.exists(config.pretrained_weights):
                model_dir = snapshot_download(config.pretrained_weights)
                pretrained_weights = os.path.join(
                    model_dir, ModelFile.TORCH_MODEL_BIN_FILE)
            elif os.path.isfile(config.pretrained_weights):
                pretrained_weights = config.pretrained_weights
            else:
                pretrained_weights = os.path.join(
                    config.pretrained_weights, ModelFile.TORCH_MODEL_BIN_FILE)
            model.load_state_dict(torch.load(pretrained_weights))
        return model


class PromptModule(nn.Module):
    """The implementation of vision prompt tuning method.

    Visual prompt tuning (VPT) is proposed to initialize tunable prompt tokens
    and prepend to the original tokens in the first layer or multiple layers.
    'Visual Prompt Tuning' by Jia et al.(2022)
    See https://arxiv.org/abs/2203.12119

    Attributes:
        dim: An integer indicating the embedding dimension.
        layer_num: An integer indicating number of layers.
        prompt_length: An integer indicating the length of vision prompt tuning.
    """

    def __init__(self,
                 dim,
                 layer_num,
                 prompt_length=None,
                 mask_values=0.,
                 attach_front=True):
        super(PromptModule, self).__init__()
        self.dim = dim
        self.layer_num = layer_num
        self.prompt_length = prompt_length
        self.mask_values = mask_values
        self.attach_front = attach_front

        self.prompt_token = nn.Parameter(torch.zeros(1, prompt_length, dim))
        nn.init.xavier_uniform_(self.prompt_token)

    def forward(self, x):
        prompt_token = self.prompt_token.expand(x.shape[0], -1, -1)

        if self.layer_num == 0:
            if self.attach_front:
                x = torch.cat((prompt_token, x), dim=1)
            else:
                x = torch.cat((x, prompt_token), dim=1)
        else:
            if self.attach_front:
                x = torch.cat((prompt_token, x[:, self.prompt_length:, :]),
                              dim=1)
            else:
                x = torch.cat((x[:, :-self.prompt_length, :], prompt_token),
                              dim=1)
        return x

    def patch_attention_mask(self, m):
        prefix_attention_mask = torch.full((*m.shape[:-1], self.prompt_length),
                                           self.mask_values).to(m.device)
        return torch.cat((prefix_attention_mask, m), dim=-1)

    def extract(self, x):
        if self.attach_front:
            return x[:, self.prompt_length:, :]
        else:
            return x[:, :-self.prompt_length, :]
