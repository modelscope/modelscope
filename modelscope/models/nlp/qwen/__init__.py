# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .configuration import QWenConfig
    from .text_generation import QWenForTextGeneration
    from .backbone import QWenModel
    from .tokenization import QWenTokenizer
else:
    _import_structure = {
        'configuration': ['QWenConfig'],
        'backbone': ['QWenModel'],
        'tokenization': ['QWenTokenizer'],
        'text_generation': ['QWenForTextGeneration'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
