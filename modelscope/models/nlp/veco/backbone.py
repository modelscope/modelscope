# Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.
# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# All rights reserved.
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
"""PyTorch Veco model. mainly copied from :module:`~transformers.modeling_xlm_roberta`"""

from transformers import RobertaModel

from modelscope.metainfo import Models
from modelscope.models import Model, TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import AttentionBackboneModelOutput
from modelscope.utils import logger as logging
from modelscope.utils.constant import Tasks
from .configuration import VecoConfig

logger = logging.get_logger()

VECO_PRETRAINED_MODEL_ARCHIVE_LIST = []


@MODELS.register_module(Tasks.backbone, module_name=Models.veco)
class VecoModel(TorchModel, RobertaModel):
    """The bare Veco Model transformer outputting raw hidden-states without any specific head on top.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config ([`VecoConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model
            weights.

    This class overrides [`RobertaModel`]. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = VecoConfig

    def __init__(self, config, **kwargs):
        super().__init__(config.name_or_path, **kwargs)
        super(Model, self).__init__(config)

    def forward(self, *args, **kwargs):
        """
        Returns:
            Returns `modelscope.outputs.AttentionBackboneModelOutputWithEmbedding`

        Examples:
            >>> from modelscope.models import Model
            >>> from modelscope.preprocessors import Preprocessor
            >>> model = Model.from_pretrained('damo/nlp_veco_fill-mask-large', task='backbone')
            >>> preprocessor = Preprocessor.from_pretrained('damo/nlp_veco_fill-mask-large')
            >>> print(model(**preprocessor('这是个测试')))

        """
        kwargs['return_dict'] = True
        outputs = super(Model, self).forward(*args, **kwargs)
        return AttentionBackboneModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=outputs.pooler_output,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    @classmethod
    def _instantiate(cls, **kwargs):
        model_dir = kwargs.pop('model_dir', None)
        if model_dir is None:
            ponet_config = VecoConfig(**kwargs)
            model = cls(ponet_config)
        else:
            model = super(
                Model,
                cls).from_pretrained(pretrained_model_name_or_path=model_dir)
        return model
