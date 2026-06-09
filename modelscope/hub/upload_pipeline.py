# Copyright (c) Alibaba, Inc. and its affiliates.
"""Upload pipeline batch tracker — shim delegating to ``modelscope_hub._upload``."""
from modelscope_hub._upload import BatchTracker  # noqa: F401

__all__ = ['BatchTracker']
