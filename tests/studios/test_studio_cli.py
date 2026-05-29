# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the ``modelscope studio`` CLI surface.

Two layers of coverage:

1. ``--help`` smoke tests via ``subprocess`` — they verify argument plumbing
   without touching the network.
2. Direct calls into :class:`~modelscope.cli.studio.StudioCMD` with
   :class:`~argparse.Namespace` instances and mocked :class:`HubApi` methods,
   which exercise the dispatch logic that ``subprocess`` cannot mock.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import unittest
from unittest.mock import ANY, patch
from uuid import uuid4

from tests.studios.conftest_env import TestResultMixin, get_test_config

from modelscope.cli.studio import StudioCMD
from modelscope.hub.api import HubApi


def _cli_invocation():
    """Return a shell-friendly invocation prefix for the modelscope CLI.

    Prefer the installed ``modelscope`` console script when available;
    fall back to ``python -c '...run_cmd()...'`` so the tests still run in
    editable installs without the entry point shim.
    """
    exe = shutil.which('modelscope')
    if exe:
        return exe
    py = sys.executable
    return (f"{py} -c 'from modelscope.cli.cli import run_cmd; run_cmd()'")


CLI_PREFIX = _cli_invocation()


class TestStudioCLIHelp(TestResultMixin, unittest.TestCase):
    """Smoke-test the help output of every studio subcommand."""

    def _run_help(self, *cli_args):
        cmd = ' '.join([CLI_PREFIX, *cli_args, '--help'])
        stat, output = subprocess.getstatusoutput(cmd)
        return stat, output

    def test_studio_help(self):
        stat, output = self._run_help('studio')
        self.assertEqual(stat, 0, output)
        for sub in ('deploy', 'stop', 'logs', 'settings', 'secret'):
            self.assertIn(sub, output)

    def test_studio_deploy_help(self):
        stat, output = self._run_help('studio', 'deploy')
        self.assertEqual(stat, 0, output)
        self.assertIn('studio_id', output)

    def test_studio_stop_help(self):
        stat, output = self._run_help('studio', 'stop')
        self.assertEqual(stat, 0, output)
        self.assertIn('studio_id', output)

    def test_studio_logs_help(self):
        stat, output = self._run_help('studio', 'logs')
        self.assertEqual(stat, 0, output)
        self.assertIn('--type', output)
        self.assertIn('--keyword', output)
        self.assertIn('--page-num', output)
        self.assertIn('--page-size', output)

    def test_studio_settings_help(self):
        stat, output = self._run_help('studio', 'settings')
        self.assertEqual(stat, 0, output)
        for flag in ('--sdk-type', '--hardware', '--private', '--public',
                     '--display-name'):
            self.assertIn(flag, output)

    def test_studio_secret_help(self):
        stat, output = self._run_help('studio', 'secret')
        self.assertEqual(stat, 0, output)
        for sub in ('list', 'add', 'update', 'delete'):
            self.assertIn(sub, output)

    def test_download_repo_type_includes_studio(self):
        stat, output = self._run_help('download')
        self.assertEqual(stat, 0, output)
        self.assertIn('studio', output)

    def test_create_includes_studio_args(self):
        stat, output = self._run_help('create')
        self.assertEqual(stat, 0, output)
        self.assertIn('--sdk-type', output)
        self.assertIn('--hardware', output)
        self.assertIn('studio', output)


class TestStudioCLIErrors(TestResultMixin, unittest.TestCase):
    """Validate user-facing error paths that don't need the network."""

    def setUp(self):
        config = get_test_config()
        self.owner = config['owner']
        # Mock tests use a fixed repo name to keep assertions deterministic.
        self.name = 'mock-studio'
        self.studio_id = f'{self.owner}/{self.name}'
        # Token from .env (MODELSCOPE_API_TOKEN); fallback for pure-mock runs.
        self.token = config['token'] or 'test-token-placeholder'

    def test_studio_deploy_missing_id(self):
        cmd = f'{CLI_PREFIX} studio deploy'
        stat, output = subprocess.getstatusoutput(cmd)
        self.assertNotEqual(stat, 0)

    def test_studio_no_subcommand_raises(self):
        args = argparse.Namespace(
            studio_action=None, token=None, endpoint=None)
        cmd = StudioCMD(args)
        with self.assertRaises(SystemExit):
            cmd.execute()

    def test_studio_settings_no_field_raises(self):
        args = argparse.Namespace(
            studio_action='settings',
            studio_id=self.studio_id,
            token=self.token,
            endpoint=None,
            display_name=None,
            description=None,
            license=None,
            cover_image=None,
            sdk_type=None,
            sdk_version=None,
            base_image=None,
            hardware=None,
            private=None,
        )
        with patch.object(
                HubApi,
                '_build_bearer_headers',
                return_value={'Authorization': f'Bearer {self.token}'}):
            cmd = StudioCMD(args)
            with self.assertRaises(SystemExit):
                cmd.execute()

    def test_secret_no_action_raises(self):
        args = argparse.Namespace(
            studio_action='secret',
            secret_action=None,
            token=None,
            endpoint=None,
        )
        with patch.object(
                HubApi,
                '_build_bearer_headers',
                return_value={'Authorization': f'Bearer {self.token}'}):
            cmd = StudioCMD(args)
            with self.assertRaises(SystemExit):
                cmd.execute()


class TestStudioCLIDirectCall(TestResultMixin, unittest.TestCase):
    """Drive ``StudioCMD.execute`` with real API calls (no HubApi mocks)."""

    @classmethod
    def setUpClass(cls):
        config = get_test_config()
        cls.token = config['token']
        cls.studio_id = config['studio_id']
        if not cls.token or not cls.studio_id:
            raise unittest.SkipTest(
                'TEST_STUDIO_ID and MODELSCOPE_API_TOKEN required '
                'for real API tests')

    def setUp(self):
        # Track secret keys created during a test for cleanup.
        self._secrets_to_cleanup = []

    def tearDown(self):
        # Best-effort cleanup of any secrets created during the test.
        api = HubApi()
        for key in self._secrets_to_cleanup:
            try:
                api.delete_studio_secret(
                    self.studio_id, key, token=self.token, endpoint=None)
            except Exception:
                pass

    def _unique_secret_key(self):
        """Generate a unique secret key with test prefix."""
        key = f'_TEST_CLI_SECRET_{uuid4().hex[:8].upper()}'
        self._secrets_to_cleanup.append(key)
        return key

    def test_deploy_direct(self):
        args = argparse.Namespace(
            studio_action='deploy',
            studio_id=self.studio_id,
            token=self.token,
            endpoint=None,
        )
        with patch('builtins.print') as mock_print:
            StudioCMD(args).execute()
        printed = ' '.join(
            str(c.args[0]) for c in mock_print.call_args_list if c.args)
        self.assertIn('Deploy triggered', printed)

    def test_stop_direct(self):
        args = argparse.Namespace(
            studio_action='stop',
            studio_id=self.studio_id,
            token=self.token,
            endpoint=None,
        )
        with patch('builtins.print') as mock_print:
            StudioCMD(args).execute()
        printed = ' '.join(
            str(c.args[0]) for c in mock_print.call_args_list if c.args)
        self.assertIn('Stop triggered', printed)

    def test_logs_direct(self):
        args = argparse.Namespace(
            studio_action='logs',
            studio_id=self.studio_id,
            token=self.token,
            endpoint=None,
            log_type='runtime',
            keyword=None,
            page_num=1,
            page_size=100,
            start_timestamp=None,
            end_timestamp=None,
        )
        # Should not raise — studio may have no logs but the call succeeds.
        with patch('builtins.print'):
            StudioCMD(args).execute()

    def test_settings_direct(self):
        display_name = f'CLI Test {int(time.time())}'
        args = argparse.Namespace(
            studio_action='settings',
            studio_id=self.studio_id,
            token=self.token,
            endpoint=None,
            display_name=display_name,
            description=None,
            license=None,
            cover_image=None,
            sdk_type=None,
            sdk_version=None,
            base_image=None,
            hardware=None,
            private=None,
        )
        with patch('builtins.print') as mock_print:
            StudioCMD(args).execute()
        printed = ' '.join(
            str(c.args[0]) for c in mock_print.call_args_list if c.args)
        self.assertIn('Updated settings', printed)

    def test_secret_list_direct(self):
        args = argparse.Namespace(
            studio_action='secret',
            secret_action='list',
            studio_id=self.studio_id,
            token=self.token,
            endpoint=None,
        )
        # Should not raise regardless of whether secrets exist.
        with patch('builtins.print'):
            StudioCMD(args).execute()

    def test_secret_add_direct(self):
        key = self._unique_secret_key()
        args = argparse.Namespace(
            studio_action='secret',
            secret_action='add',
            studio_id=self.studio_id,
            key=key,
            value='test_value',
            token=self.token,
            endpoint=None,
        )
        with patch('builtins.print') as mock_print:
            StudioCMD(args).execute()
        printed = ' '.join(
            str(c.args[0]) for c in mock_print.call_args_list if c.args)
        self.assertIn('added', printed.lower())

    def test_secret_update_direct(self):
        key = self._unique_secret_key()
        # Pre-add the secret so we can update it.
        api = HubApi()
        api.add_studio_secret(
            self.studio_id,
            key,
            'initial_value',
            token=self.token,
            endpoint=None)

        args = argparse.Namespace(
            studio_action='secret',
            secret_action='update',
            studio_id=self.studio_id,
            key=key,
            value='updated_value',
            token=self.token,
            endpoint=None,
        )
        with patch('builtins.print') as mock_print:
            StudioCMD(args).execute()
        printed = ' '.join(
            str(c.args[0]) for c in mock_print.call_args_list if c.args)
        self.assertIn('updated', printed.lower())

    def test_secret_delete_direct(self):
        key = self._unique_secret_key()
        # Pre-add the secret so we can delete it.
        api = HubApi()
        api.add_studio_secret(
            self.studio_id,
            key,
            'to_be_deleted',
            token=self.token,
            endpoint=None)
        # Remove from cleanup list since we expect CLI to delete it.
        self._secrets_to_cleanup.remove(key)

        args = argparse.Namespace(
            studio_action='secret',
            secret_action='delete',
            studio_id=self.studio_id,
            key=key,
            token=self.token,
            endpoint=None,
        )
        with patch('builtins.print') as mock_print:
            StudioCMD(args).execute()
        printed = ' '.join(
            str(c.args[0]) for c in mock_print.call_args_list if c.args)
        self.assertIn('deleted', printed.lower())


if __name__ == '__main__':
    unittest.main()
