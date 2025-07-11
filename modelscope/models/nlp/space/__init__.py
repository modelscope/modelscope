# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .configuration import SpaceConfig
    from .dialog_intent_prediction import SpaceForDialogIntent
    from .dialog_modeling import SpaceForDialogModeling
    from .dialog_state_tracking import SpaceForDST
    from .model import SpaceGenerator, SpaceModelBase, SpaceTokenizer
else:
    _import_structure = {
        'model': ['SpaceGenerator', 'SpaceModelBase', 'SpaceTokenizer'],
        'dialog_intent_prediction': ['SpaceForDialogIntent'],
        'dialog_modeling': ['SpaceForDialogModeling'],
        'dialog_state_tracking': ['SpaceForDST'],
        'configuration': ['SpaceConfig']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
