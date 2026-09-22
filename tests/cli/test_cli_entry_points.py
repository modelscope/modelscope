# Copyright (c) Alibaba, Inc. and its affiliates.
"""Guards for CLI console-script ownership.

Every ModelScope console script (``modelscope``, ``ms``, ``modelscope-hub``,
``ms-hub``) is installed by the ``modelscope-hub`` distribution. If this package
declared any of them again, the two distributions would fight over the same
file: upgrading or uninstalling either one could delete the other's script and
leave the user with no CLI, and OS packagers such as FreeBSD pkg refuse to let
two ports own one path. These tests keep that arrangement from regressing
silently.
"""

import subprocess
import sys
import unittest
from pathlib import Path

PLUGIN_GROUP = 'modelscope_hub.cli_plugins'
HUB_DIST = 'modelscope-hub'
REPO_ROOT = Path(__file__).resolve().parents[2]


def _pyproject():
    """Parse the project metadata, or skip where tomllib is unavailable."""
    if sys.version_info < (3, 11):
        # tomllib is stdlib from 3.11; the declaration it reads is not
        # interpreter-specific, so checking on newer runtimes is enough.
        raise unittest.SkipTest('stdlib tomllib requires Python 3.11+')
    import tomllib
    return tomllib.loads(
        (REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))


class TestConsoleScriptOwnership(unittest.TestCase):
    """This package must not declare console scripts of its own."""

    def test_declares_no_console_scripts(self):
        scripts = _pyproject()['project'].get('scripts', {})
        self.assertEqual(
            scripts, {},
            'Console scripts belong to the modelscope-hub distribution. '
            'Declaring them here lets either package delete the other side.')

    def test_contributes_commands_as_plugins(self):
        """Giving up the scripts only works if the plugin group survives."""
        entry_points = _pyproject()['project']['entry-points']
        self.assertIn(PLUGIN_GROUP, entry_points)
        self.assertTrue(entry_points[PLUGIN_GROUP])

    def test_depends_on_the_script_owner(self):
        """The CLI now comes from modelscope-hub, so it cannot be optional."""
        requirements = (REPO_ROOT / 'requirements'
                        / 'hub.txt').read_text(encoding='utf-8')
        self.assertIn(HUB_DIST, requirements)


class TestCliShim(unittest.TestCase):
    """``python -m modelscope.cli.cli`` must survive the handover."""

    def test_delegates_to_the_hub_engine(self):
        from modelscope_hub.cli.main import run_cmd as hub_run_cmd

        from modelscope.cli import cli
        self.assertIs(cli._run_cmd, hub_run_cmd)

    def test_module_entry_point_still_runs(self):
        result = subprocess.run(
            [sys.executable, '-m', 'modelscope.cli.cli', '--version'],
            capture_output=True,
            text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('modelscope', result.stdout)


if __name__ == '__main__':
    unittest.main()
