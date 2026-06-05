# Copyright (c) Alibaba, Inc. and its affiliates.
"""Upload hash cache — shim delegating to ``modelscope_hub._upload``.

The unified :class:`modelscope_hub._upload.UploadTracker` supersedes
this module's previous standalone hash cache; it remains here for any
caller that still imports the legacy file constant.
"""
from modelscope_hub._upload import UploadTracker as UploadHashCache  # noqa: F401
from modelscope_hub.constants import UPLOAD_CACHE_FILE as UPLOAD_HASH_CACHE_FILE  # noqa: F401

__all__ = ['UploadHashCache', 'UPLOAD_HASH_CACHE_FILE']
