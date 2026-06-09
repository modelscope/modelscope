# Copyright (c) Alibaba, Inc. and its affiliates.
"""ModelScope CLI — delegates to the modelscope_hub CLI engine.

The legacy ``modelscope`` / ``ms`` console-script entry points historically
lived here as a hand-rolled argparse tree. The hub CLI in ``modelscope_hub``
now owns command registration, plugin discovery, and error translation;
this module exists solely to preserve the import path used by the
``[project.scripts]`` entries in ``pyproject.toml``.
"""

import sys

from modelscope_hub.cli.main import run_cmd as _run_cmd


def run_cmd():
    """Delegate to ``modelscope_hub.cli.main.run_cmd`` and propagate its exit code."""
    sys.exit(_run_cmd())


if __name__ == '__main__':
    run_cmd()
