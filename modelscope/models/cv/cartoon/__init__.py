# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .facelib.facer import FaceAna
    from .mtcnn_pytorch.src.align_trans import (get_reference_facial_points,
                                                warp_and_crop_face)
    from .utils import (get_f5p, padTo16x, resize_size, all_file,
                        tf_data_loader, write_batch_image)
    from .network import disc_sn
    from .loss import simple_superpixel
    from .model_tf import CartoonModel

else:
    _import_structure = {
        'facelib.facer': ['FaceAna'],
        'mtcnn_pytorch.src.align_trans':
        ['get_reference_facial_points', 'warp_and_crop_face'],
        'utils': [
            'get_f5p', 'padTo16x', 'resize_size', 'all_file', 'tf_data_loader',
            'write_batch_image'
        ],
        'network': ['disc_sn'],
        'loss': ['simple_superpixel'],
        'model_tf': ['CartoonModel'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
