# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch LLaMA model."""
from transformers.models.llama import LlamaConfig
from transformers.models.llama import LlamaModel as LlamaModelHF
from transformers.models.llama import \
    LlamaPreTrainedModel as LlamaPreTrainedModelHF

from modelscope.metainfo import Models
from modelscope.models import Model, TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


class MsModelMixin:

    @classmethod
    def _instantiate(cls, **kwargs):
        """Instantiate the model.

        Args:
            kwargs: Input args.
                    model_dir: The model dir used to load the checkpoint and the label information.
                    num_labels: An optional arg to tell the model how many classes to initialize.
                                    Method will call utils.parse_label_mapping if num_labels not supplied.
                                    If num_labels is not found, the model will use the default setting (2 classes).

        Returns:
            The loaded model, which is initialized by transformers.PreTrainedModel.from_pretrained
        """
        model_dir = kwargs.pop('model_dir', None)
        device = kwargs.pop('device', None)
        if model_dir is None:
            config = LlamaConfig(**kwargs)
            model = cls(config)
        else:
            model = super(MsModelMixin, cls).from_pretrained(
                pretrained_model_name_or_path=model_dir, **kwargs)
        model.model_dir = model_dir
        return model if 'device_map' in kwargs \
            or device is None else model.to(device)


class LlamaPreTrainedModel(MsModelMixin, LlamaPreTrainedModelHF, TorchModel):
    pass


@MODELS.register_module(Tasks.backbone, module_name=Models.llama2)
@MODELS.register_module(Tasks.backbone, module_name=Models.llama)
class LlamaModel(MsModelMixin, LlamaModelHF, TorchModel):
    pass
