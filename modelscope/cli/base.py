# Copyright (c) Alibaba, Inc. and its affiliates.
"""CLI base class — re-exports :class:`CLICommand` from ``modelscope_hub``.

Kept as a thin alias so existing imports such as
``from modelscope.cli.base import CLICommand`` continue to work after the
CLI engine moved into ``modelscope_hub``.
"""

from modelscope_hub.cli.base import CLICommand  # noqa: F401

__all__ = ['CLICommand']
