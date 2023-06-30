import inspect
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
class AdapterConfig(SwiftConfig):
    """
    The configuration class for the adapter module.

    Adapters project input tokens by an MLP layer.
    'Parameter-Efficient Transfer Learning for NLP' by Houlsby et al.(2019)
    See http://arxiv.org/abs/1902.00751

    Args:
        dim: The dimension of the hidden states
        module_name: The feedforward module to be replaced, in regex format
        hidden_pos: The position of the hidden state to passed into the adapter, can be int (args) or str (kwargs)
        method_name: The method to be replaced, default to replace the forward method
        adapter_length: The length of the adapter length (intermediate length)
        act_layer: The activation layer of the adapter
        only_adapter_trainable: Whether to train only adapters
        pretrained_weights: The pretrained adapter weights.
            Can be a local dir, local file, or a model id from modelscope
    """

    dim: int = field(metadata={'help': 'The dimension of the hidden states'})

    module_name: str = field(
        metadata={
            'help': 'The feedforward module to be replaced, in regex format'
        })

    hidden_pos: Union[str, int] = field(
        metadata={
            'help':
            'The position of the hidden state to passed into the adapter, can be int (args) or str (kwargs)'
        })

    method_name: str = field(
        default='forward',
        metadata={
            'help':
            'The method to be replaced, default to replace the forward method'
        })

    adapter_length: int = field(
        default=128,
        metadata={
            'help': 'The length of the adapter length (intermediate length)'
        })

    act_layer: nn.Module = field(
        default=nn.GELU,
        metadata={'help': 'The activation layer of the adapter'})

    only_adapter_trainable: bool = field(
        default=True, metadata={'help': 'Whether to train only adapters'})

    pretrained_weights: str = field(
        default=None,
        metadata={
            'help':
            'The pretrained adapter weights. Can be a local dir, local file, or a model id from modelscope'
        })


class Adapter:

    @staticmethod
    def prepare_model(model: nn.Module, config: AdapterConfig):
        module_keys = [key for key, _ in model.named_modules()]

        for module_key in module_keys:
            if re.fullmatch(config.module_name, module_key):  # noqa
                module = model.get_submodule(module_key)

                def _forward(self, *args, **kwargs):
                    args = self.forward_origin(*args, **kwargs)
                    if isinstance(args, (tuple, list, dict)):
                        if isinstance(config.hidden_pos, int):
                            return args[0:config.hidden_pos] + args[
                                config.hidden_pos] + getattr(self, 'adapter')(args[config.hidden_pos]) \
                                + args[config.hidden_pos + 1:] # noqa
                        else:
                            kwargs[config.hidden_pos] = args[
                                config.hidden_pos] + getattr(self, 'adapter')(
                                    args[config.hidden_pos])
                    elif isinstance(args, torch.Tensor):
                        args = getattr(self, 'adapter')(args)
                    return args

                def _feed_forward_chunk(self, attention_output):
                    return _forward(self, attention_output)

                module.forward_origin = getattr(module, config.method_name)
                num_args_in_forward_chunk_fn = len(
                    inspect.signature(module.forward_origin).parameters)
                if config.method_name == 'feed_forward_chunk' and num_args_in_forward_chunk_fn == 1:
                    setattr(module, config.method_name,
                            types.MethodType(_feed_forward_chunk, module))
                else:
                    setattr(module, config.method_name,
                            types.MethodType(_forward, module))
                adapter_module = AdapterModule(config.dim,
                                               config.adapter_length,
                                               config.act_layer)
                setattr(module, 'adapter', adapter_module)

        if config.only_adapter_trainable:
            for n, p in model.named_parameters():
                if 'adapter' not in n:
                    p.requires_grad = False

        def state_dict_hook(module, destination, prefix, local_metadata):
            return {
                key: value
                for key, value in destination.items() if 'adapter' in key
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


class AdapterModule(nn.Module):
    """The implementation of adapter tuning method.

    Adapters project input tokens by an MLP layer.
    'Parameter-Efficient Transfer Learning for NLP' by Houlsby et al.(2019)
    See http://arxiv.org/abs/1902.00751

    Attributes:
        dim: An integer indicating the embedding dimension.
        adapter_length: An integer indicating the length of adapter tuning.
    """

    def __init__(
        self,
        dim,
        adapter_length=None,
        act_layer=nn.GELU,
    ):
        super(AdapterModule, self).__init__()
        self.dim = dim
        self.adapter_length = adapter_length
        # self.adapter_type = adapter_type
        self.ln1 = nn.Linear(dim, adapter_length)
        self.activate = act_layer()
        self.ln2 = nn.Linear(adapter_length, dim)
        self.init_weights()

    def init_weights(self):

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

        self.apply(_init_weights)

    def forward(self, x, identity=None):
        out = self.ln2(self.activate(self.ln1(x)))
        if identity is None:
            identity = x
        out = identity + out
        return out
