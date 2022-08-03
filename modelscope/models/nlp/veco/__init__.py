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

from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .configuration_veco import VecoConfig
    from .modeling_veco import (VecoForMaskedLM, VecoForSequenceClassification,
                                VecoModel)
    from .tokenization_veco import VecoTokenizer
    from .tokenization_veco_fast import VecoTokenizerFast
else:
    _import_structure = {
        'configuration_veco': ['VecoConfig'],
        'modeling_veco':
        ['VecoForMaskedLM', 'VecoForSequenceClassification', 'VecoModel'],
        'tokenization_veco': ['VecoTokenizer'],
        'tokenization_veco_fast': ['VecoTokenizerFast'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
