# Copyright (c) Alibaba, Inc. and its affiliates.
"""ModelScope CLI — delegates to the modelscope_hub CLI engine.

The ``modelscope`` / ``ms`` console scripts historically lived here as a
hand-rolled argparse tree. ``modelscope_hub`` now owns command registration,
plugin discovery, error translation *and* the console-script declarations for
all four aliases (``modelscope``, ``ms``, ``modelscope-hub``, ``ms-hub``), so
that a single distribution writes those files and neither package can strand
the other's CLI on upgrade.

This module stays as the ``python -m modelscope.cli.cli`` entry point and as a
stable import path for callers that reference it directly. The commands this
package adds on top of the hub CLI are contributed through the
``modelscope_hub.cli_plugins`` entry-point group declared in ``pyproject.toml``.
"""

import sys

from modelscope_hub.cli.main import run_cmd as _run_cmd


def run_cmd():
    """Delegate to ``modelscope_hub.cli.main.run_cmd`` and propagate its exit code."""
    sys.exit(_run_cmd())


if __name__ == '__main__':
    run_cmd()
