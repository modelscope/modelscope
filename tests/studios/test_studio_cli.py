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

from tests.studios.conftest_env import (TestResultMixin, create_temp_studio,
                                         get_test_config)

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


class TestStudioCreate(TestResultMixin, unittest.TestCase):
    """Test studio creation via HubApi.create_repo(repo_type='studio')."""

    @classmethod
    def setUpClass(cls):
        config = get_test_config()
        cls.token = config['token']
        cls.owner = config['owner']
        cls.config = config
        cls._cleanup = config.get('cleanup', True)
        if not cls.token or not cls.owner:
            raise unittest.SkipTest(
                'MODELSCOPE_API_TOKEN and TEST_STUDIO_OWNER required')
        cls._created_studios = []

    @classmethod
    def tearDownClass(cls):
        """Studio deletion is not supported via OpenAPI. Log created studios for manual cleanup."""
        if cls._created_studios:
            import logging
            logging.getLogger('modelscope').info(
                f'Test studios created (manual cleanup needed): '
                f'{", ".join(cls._created_studios)}')

    def _create_and_track(self, **kwargs):
        """Create a studio with unique name, track for cleanup."""
        name = f'ut_test_create_{uuid4().hex[:8]}'
        repo_id = f'{self.owner}/{name}'
        api = HubApi()
        url = api.create_repo(
            repo_id,
            repo_type='studio',
            visibility=self.config.get('visibility', 'private'),
            token=self.token,
            endpoint=self.config.get('endpoint'),
            create_default_config=False,
            **kwargs,
        )
        self._created_studios.append(repo_id)
        return repo_id, url

    def test_create_studio_basic(self):
        """Create a basic studio and verify it exists."""
        repo_id, url = self._create_and_track(sdk_type='gradio')
        self.assertIn(repo_id, url)
        # Verify it actually exists
        api = HubApi()
        exists = api.repo_exists(
            repo_id, repo_type='studio', token=self.token)
        self.assertTrue(exists,
                        f'Studio {repo_id} should exist after creation')

    def test_create_studio_exist_ok(self):
        """Creating an existing studio with exist_ok=True should not raise."""
        repo_id, _ = self._create_and_track(sdk_type='gradio')
        api = HubApi()
        # Create again with exist_ok=True
        url = api.create_repo(
            repo_id,
            repo_type='studio',
            visibility=self.config.get('visibility', 'private'),
            token=self.token,
            endpoint=self.config.get('endpoint'),
            exist_ok=True,
            create_default_config=False,
        )
        self.assertIn(repo_id, url)

    def test_create_studio_exist_raises(self):
        """Creating an existing studio without exist_ok should raise ValueError."""
        repo_id, _ = self._create_and_track(sdk_type='gradio')
        api = HubApi()
        with self.assertRaises(ValueError):
            api.create_repo(
                repo_id,
                repo_type='studio',
                visibility=self.config.get('visibility', 'private'),
                token=self.token,
                endpoint=self.config.get('endpoint'),
                exist_ok=False,
                create_default_config=False,
            )


class TestStudioCLIDirectCall(TestResultMixin, unittest.TestCase):
    """Drive ``StudioCMD.execute`` with real API calls (no HubApi mocks)."""

    @classmethod
    def setUpClass(cls):
        config = get_test_config()
        cls.token = config['token']
        cls.studio_id = config['studio_id']
        cls._auto_created_studio = False
        cls._cleanup = config.get('cleanup', True)

        if not cls.token:
            raise unittest.SkipTest(
                'MODELSCOPE_API_TOKEN required for real API tests')

        # If no TEST_STUDIO_ID configured, create a temporary one
        if not cls.studio_id:
            cls.studio_id = create_temp_studio(config)
            cls._auto_created_studio = True

    @classmethod
    def tearDownClass(cls):
        """Studio deletion is not supported via OpenAPI. Log for manual cleanup."""
        if cls._auto_created_studio and cls.studio_id:
            import logging
            logging.getLogger('modelscope').info(
                f'Auto-created test studio (manual cleanup needed): '
                f'{cls.studio_id}')

    def setUp(self):
        # Track secret keys created during a test for cleanup.
        self._secrets_to_cleanup = []

    def tearDown(self):
        # Best-effort cleanup of any secrets created during the test.
        if not self._cleanup:
            return
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
