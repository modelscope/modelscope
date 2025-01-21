# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .cartoon_translation_exporter import CartoonTranslationExporter
    from .face_detection_scrfd_exporter import FaceDetectionSCRFDExporter
    from .object_detection_damoyolo_exporter import \
        ObjectDetectionDamoyoloExporter
else:
    _import_structure = {
        'cartoon_translation_exporter': ['CartoonTranslationExporter'],
        'object_detection_damoyolo_exporter':
        ['ObjectDetectionDamoyoloExporter'],
        'face_detection_scrfd_exporter': ['FaceDetectionSCRFDExporter'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
