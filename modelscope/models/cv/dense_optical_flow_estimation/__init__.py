from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .raft_model import DenseOpticalFlowEstimation

else:
    _import_structure = {
        'raft_dense_optical_flow_estimation': ['DenseOpticalFlowEstimation'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
