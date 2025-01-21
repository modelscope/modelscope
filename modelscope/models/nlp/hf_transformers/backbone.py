# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model. """

from transformers import AutoConfig, AutoModel
from transformers.modeling_utils import PreTrainedModel

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import AttentionBackboneModelOutput
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.nlp.utils import parse_labels_in_order

logger = get_logger()


def _get_model_class(config, model_mapping):
    supported_models = model_mapping[type(config)]
    if not isinstance(supported_models, (list, tuple)):
        return supported_models

    name_to_model = {model.__name__: model for model in supported_models}
    architectures = getattr(config, 'architectures', [])
    for arch in architectures:
        if arch in name_to_model:
            return name_to_model[arch]
        elif f'TF{arch}' in name_to_model:
            return name_to_model[f'TF{arch}']
        elif f'Flax{arch}' in name_to_model:
            return name_to_model[f'Flax{arch}']

    # If not architecture is set in the config or match the supported models, the first element of the tuple is the
    # defaults.
    return supported_models[0]


@MODELS.register_module(
    group_key=Tasks.backbone, module_name=Models.transformers)
class TransformersModel(TorchModel, PreTrainedModel):
    """The Bert Model transformer outputting raw hidden-states without any
    specific head on top.

    This model inherits from [`PreTrainedModel`]. Check the superclass
    documentation for the generic methods the library implements for all its
    model (such as downloading or saving, resizing the input embeddings, pruning
    heads etc.)

    This model is also a PyTorch
    [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch
    documentation for all matter related to general usage and behavior.

    Parameters:
        config ([`BertConfig`]): Model configuration class with all the
        parameters of the model.
            Initializing with a config file does not load the weights associated
            with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model
            weights.

    The model can behave as an encoder (with only self-attention) as well as a
    decoder, in which case a layer of cross-attention is added between the
    self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam
    Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    `is_decoder` argument of the configuration set to `True`. To be used in a
    Seq2Seq model, the model needs to initialized with both `is_decoder`
    argument and `add_cross_attention` set to `True`; an `encoder_hidden_states`
    is then expected as an input to the forward pass.


    """

    @classmethod
    def _instantiate(cls, model_dir=None, **config):
        init_backbone = config.pop('init_backbone', False)

        # return the model with pretrained weights
        if init_backbone:
            model = AutoModel.from_pretrained(model_dir)
            return model

        # return the model only
        config, kwargs = AutoConfig.from_pretrained(
            model_dir,
            return_unused_kwargs=True,
            trust_remote_code=False,
            **config)

        model_mapping = AutoModel._model_mapping
        if type(config) in model_mapping.keys():
            model_class = _get_model_class(config, model_mapping)
            model = model_class(config)
            model.model_dir = model_dir
            return model

        raise ValueError(
            f'Unrecognized configuration class {config.__class__} for the AutoModel'
            f"Model type should be one of {', '.join(c.__name__ for c in model_mapping.keys())}."
        )
