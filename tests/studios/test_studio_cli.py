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
import unittest
from unittest.mock import ANY, patch

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


class TestStudioCLIHelp(unittest.TestCase):
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


class TestStudioCLIErrors(unittest.TestCase):
    """Validate user-facing error paths that don't need the network."""

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
            studio_id='owner/name',
            token='fake-token',
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
                return_value={'Authorization': 'Bearer fake'}):
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
                return_value={'Authorization': 'Bearer fake'}):
            cmd = StudioCMD(args)
            with self.assertRaises(SystemExit):
                cmd.execute()


class TestStudioCLIDirectCall(unittest.TestCase):
    """Directly drive ``StudioCMD.execute`` with mocked ``HubApi`` methods."""

    @patch.object(HubApi, 'deploy_studio')
    @patch.object(HubApi, '_build_bearer_headers')
    def test_deploy_direct(self, mock_headers, mock_deploy):
        mock_headers.return_value = {'Authorization': 'Bearer fake'}
        mock_deploy.return_value = {
            'status': 'Deploying',
            'active_config': {
                'hardware': 'platform/2v-cpu-16g-mem'
            },
        }
        args = argparse.Namespace(
            studio_action='deploy',
            studio_id='testuser/my-app',
            token='fake-token',
            endpoint=None,
        )
        with patch('builtins.print'):
            StudioCMD(args).execute()
        mock_deploy.assert_called_once()
        call_args, call_kwargs = mock_deploy.call_args
        self.assertEqual(call_args[0], 'testuser/my-app')
        self.assertEqual(call_kwargs.get('token'), 'fake-token')

    @patch.object(HubApi, 'stop_studio')
    @patch.object(HubApi, '_build_bearer_headers')
    def test_stop_direct(self, mock_headers, mock_stop):
        mock_headers.return_value = {'Authorization': 'Bearer fake'}
        mock_stop.return_value = {'status': 'Stopping'}
        args = argparse.Namespace(
            studio_action='stop',
            studio_id='testuser/my-app',
            token='fake-token',
            endpoint=None,
        )
        with patch('builtins.print'):
            StudioCMD(args).execute()
        mock_stop.assert_called_once()
        self.assertEqual(mock_stop.call_args[0][0], 'testuser/my-app')

    @patch.object(HubApi, 'get_studio_logs')
    @patch.object(HubApi, '_build_bearer_headers')
    def test_logs_direct(self, mock_headers, mock_logs):
        mock_headers.return_value = {'Authorization': 'Bearer fake'}
        mock_logs.return_value = {
            'logs': ['line1', 'line2'],
            'total': 2,
        }
        args = argparse.Namespace(
            studio_action='logs',
            studio_id='testuser/my-app',
            token='fake-token',
            endpoint=None,
            log_type='runtime',
            keyword=None,
            page_num=1,
            page_size=100,
            start_timestamp=None,
            end_timestamp=None,
        )
        with patch('builtins.print'):
            StudioCMD(args).execute()
        mock_logs.assert_called_once()
        kwargs = mock_logs.call_args.kwargs
        self.assertEqual(kwargs['log_type'], 'runtime')
        self.assertEqual(kwargs['page_num'], 1)
        self.assertEqual(kwargs['page_size'], 100)
        self.assertEqual(kwargs['keyword'], None)

    @patch.object(HubApi, 'update_studio_settings')
    @patch.object(HubApi, '_build_bearer_headers')
    def test_settings_direct(self, mock_headers, mock_settings):
        mock_headers.return_value = {'Authorization': 'Bearer fake'}
        mock_settings.return_value = {'display_name': 'Updated'}
        args = argparse.Namespace(
            studio_action='settings',
            studio_id='testuser/my-app',
            token='fake-token',
            endpoint=None,
            display_name='Updated',
            description=None,
            license=None,
            cover_image=None,
            sdk_type=None,
            sdk_version=None,
            base_image=None,
            hardware=None,
            private=None,
        )
        with patch('builtins.print'):
            StudioCMD(args).execute()
        mock_settings.assert_called_once()
        call_kwargs = mock_settings.call_args.kwargs
        self.assertEqual(call_kwargs.get('display_name'), 'Updated')
        # Only specified fields are forwarded.
        self.assertNotIn('sdk_type', call_kwargs)
        self.assertNotIn('hardware', call_kwargs)

    @patch.object(HubApi, 'list_studio_secrets')
    @patch.object(HubApi, '_build_bearer_headers')
    def test_secret_list_direct(self, mock_headers, mock_list):
        mock_headers.return_value = {'Authorization': 'Bearer fake'}
        mock_list.return_value = [{'key': 'API_KEY'}, {'key': 'SECRET'}]
        args = argparse.Namespace(
            studio_action='secret',
            secret_action='list',
            studio_id='testuser/my-app',
            token='fake-token',
            endpoint=None,
        )
        with patch('builtins.print') as mock_print:
            StudioCMD(args).execute()
        mock_list.assert_called_once()
        printed = ' '.join(
            str(c.args[0]) for c in mock_print.call_args_list if c.args)
        self.assertIn('API_KEY', printed)

    @patch.object(HubApi, 'list_studio_secrets')
    @patch.object(HubApi, '_build_bearer_headers')
    def test_secret_list_empty_direct(self, mock_headers, mock_list):
        mock_headers.return_value = {'Authorization': 'Bearer fake'}
        mock_list.return_value = []
        args = argparse.Namespace(
            studio_action='secret',
            secret_action='list',
            studio_id='testuser/my-app',
            token=None,
            endpoint=None,
        )
        with patch('builtins.print') as mock_print:
            StudioCMD(args).execute()
        printed = ' '.join(
            str(c.args[0]) for c in mock_print.call_args_list if c.args)
        self.assertIn('no secrets', printed)

    @patch.object(HubApi, 'add_studio_secret')
    @patch.object(HubApi, '_build_bearer_headers')
    def test_secret_add_direct(self, mock_headers, mock_add):
        mock_headers.return_value = {'Authorization': 'Bearer fake'}
        args = argparse.Namespace(
            studio_action='secret',
            secret_action='add',
            studio_id='testuser/my-app',
            key='MY_KEY',
            value='my_value',
            token='fake-token',
            endpoint=None,
        )
        with patch('builtins.print'):
            StudioCMD(args).execute()
        mock_add.assert_called_once_with(
            'testuser/my-app',
            'MY_KEY',
            'my_value',
            token='fake-token',
            endpoint=ANY)

    @patch.object(HubApi, 'update_studio_secret')
    @patch.object(HubApi, '_build_bearer_headers')
    def test_secret_update_direct(self, mock_headers, mock_update):
        mock_headers.return_value = {'Authorization': 'Bearer fake'}
        args = argparse.Namespace(
            studio_action='secret',
            secret_action='update',
            studio_id='testuser/my-app',
            key='MY_KEY',
            value='new_value',
            token='fake-token',
            endpoint=None,
        )
        with patch('builtins.print'):
            StudioCMD(args).execute()
        mock_update.assert_called_once_with(
            'testuser/my-app',
            'MY_KEY',
            'new_value',
            token='fake-token',
            endpoint=ANY)

    @patch.object(HubApi, 'delete_studio_secret')
    @patch.object(HubApi, '_build_bearer_headers')
    def test_secret_delete_direct(self, mock_headers, mock_delete):
        mock_headers.return_value = {'Authorization': 'Bearer fake'}
        args = argparse.Namespace(
            studio_action='secret',
            secret_action='delete',
            studio_id='testuser/my-app',
            key='MY_KEY',
            token='fake-token',
            endpoint=None,
        )
        with patch('builtins.print'):
            StudioCMD(args).execute()
        mock_delete.assert_called_once_with(
            'testuser/my-app', 'MY_KEY', token='fake-token', endpoint=ANY)


if __name__ == '__main__':
    unittest.main()
