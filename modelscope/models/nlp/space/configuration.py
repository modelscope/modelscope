# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# Copyright 2018 The Google AI Language Team Authors.
# Copyright 2020 The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.
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
"""Space configuration, mainly copied from :class:`~transformers.configuration_xlm_roberta` """

from modelscope.models.nlp.structbert import SbertConfig
from modelscope.utils import logger as logging

logger = logging.get_logger()


class SpaceConfig(SbertConfig):
    """
    This class overrides [`SbertConfig`]. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    model_type = 'space'
