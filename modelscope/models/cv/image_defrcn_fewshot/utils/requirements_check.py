# Copyright (c) Alibaba, Inc. and its affiliates.

import importlib
import sys
from collections import OrderedDict

from packaging import version

from modelscope.utils.import_utils import _torch_available

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

DETECTRON2_REQUIRED_VERSION = version.parse('0.3')


def is_detectron2_version_available():
    _detectron2_available = importlib.util.find_spec('detectron2') is not None
    _detectron2_version_available = False
    if _detectron2_available:
        _detectron2_version = version.parse(
            importlib_metadata.version('detectron2'))
        _detectron2_version_available = (_detectron2_version.major,
                                         _detectron2_version.minor) == (
                                             DETECTRON2_REQUIRED_VERSION.major,
                                             DETECTRON2_REQUIRED_VERSION.minor)

    return _detectron2_version_available


TORCH_REQUIRED_VERSION = version.parse('1.11')


def is_torch_version_available():
    _torch_version_available = False
    if _torch_available:
        torch_version = version.parse(importlib_metadata.version('torch'))
        _torch_version_available = (torch_version.major,
                                    torch_version.minor) == (
                                        TORCH_REQUIRED_VERSION.major,
                                        TORCH_REQUIRED_VERSION.minor)

    return _torch_version_available


DETECTRON2_IMPORT_ERROR = """
{0} requires the detectron2-0.3 but it was not found in your environment.
You can install it from modelscope lib with pip:
`pip install detectron2==0.3`
"""

TORCH_VERSION_IMPORT_ERROR = """
{0} requires the torch-1.11 but it was not found in your environment. You can install it with pip:
`pip install torch==1.11`
"""

REQUIREMENTS_MAAPING_VERSION = OrderedDict([
    ('detectron2-0.3', (is_detectron2_version_available,
                        DETECTRON2_IMPORT_ERROR)),
    ('torch-1.11', (is_torch_version_available, TORCH_VERSION_IMPORT_ERROR)),
])

REQUIREMENTS = ['detectron2-0.3', 'torch-1.11']


def requires_version():
    checks = []
    for req in REQUIREMENTS:
        if req in REQUIREMENTS_MAAPING_VERSION:
            check = REQUIREMENTS_MAAPING_VERSION[req]
        else:
            raise NotImplementedError('{} do not supported check'.format(req))
        checks.append(check)

    failed = [
        msg.format('DeFRCN') for available, msg in checks if not available()
    ]
    if failed:
        raise ImportError(''.join(failed))
