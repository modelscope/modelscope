# The implementation is made publicly available under the
# Apache 2.0 license at https://github.com/cvg/LightGlue

from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .lightglue_model import LightGlueImageMatching

else:
    _import_structure = {
        'lightglue_model': ['LightGlueImageMatching'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
