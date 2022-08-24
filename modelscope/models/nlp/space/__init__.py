from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .model import SpaceGenerator
    from .model import SpaceModelBase, SpaceTokenizer, SpaceConfig
    from .space_for_dialog_intent_prediction import SpaceForDialogIntent
    from .space_for_dialog_modeling import SpaceForDialogModeling
    from .space_for_dialog_state_tracking import SpaceForDialogStateTracking
else:
    _import_structure = {
        'model':
        ['SpaceGenerator', 'SpaceModelBase', 'SpaceTokenizer', 'SpaceConfig'],
        'space_for_dialog_intent_prediction': ['SpaceForDialogIntent'],
        'space_for_dialog_modeling': ['SpaceForDialogModeling'],
        'space_for_dialog_state_tracking': ['SpaceForDialogStateTracking'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
