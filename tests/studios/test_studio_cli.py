# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the ``modelscope studio`` CLI surface.

Two layers of coverage:

1. ``--help`` smoke tests via ``subprocess`` — they verify argument plumbing
   without touching the network.
2. Direct calls into :class:`~modelscope.cli.studio.StudioCMD` with
   :class:`argparse.Namespace` instances — these exercise the dispatch
   logic against the real API (``print`` is mocked for output capture).

Secret/variable ``description`` coverage is gated on
:func:`_hub_supports_description` and the Studio OAuth app configuration on
:func:`_hub_supports_studio_oauth`, because the modelscope_hub floor pinned
in requirements/hub.txt predates both features; the gates disappear from the
reports once an adapted hub is installed.
"""
import argparse
import inspect
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


def _hub_supports_description():
    """True when the installed modelscope_hub forwards secret/variable
    ``description`` fields.

    The description support landed in modelscope_hub after the 0.4.6 release
    currently pinned as the floor in requirements/hub.txt. Guard the
    new-surface tests so this suite still passes against the released floor
    while exercising the full surface once an adapted hub is installed.
    """
    try:
        from modelscope_hub.api import HubApi as _HubApi
    except ImportError:  # pragma: no cover
        return False
    try:
        return 'description' in inspect.signature(
            _HubApi.add_secret).parameters
    except (AttributeError, TypeError):  # pragma: no cover
        return False


def _hub_supports_studio_oauth():
    """True when the installed modelscope_hub exposes the Studio OAuth app
    configuration flags (``--oauth-enabled`` and friends on
    ``studio settings`` / ``create``).

    The Studio OAuth surface comes from the modelscope_service
    ``feat/openAPI_studios_ouath`` branch and is adapted in modelscope_hub
    after the 0.4.6 release floor; guard the new tests accordingly.
    """
    try:
        from modelscope_hub.cli import studio as _hub_studio_cli
    except ImportError:  # pragma: no cover
        return False
    return 'oauth_enabled' in getattr(_hub_studio_cli, '_SETTINGS_FIELDS', ())


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
    """Smoke-test the help output of studio-related subcommands.

    In the new CLI engine, studio operations are top-level commands
    (deploy, stop, logs, settings, secret) rather than nested under
    a ``studio`` group.
    """

    def _run_help(self, *cli_args):
        cmd = ' '.join([CLI_PREFIX, *cli_args, '--help'])
        stat, output = subprocess.getstatusoutput(cmd)
        return stat, output

    def test_studio_help(self):
        """The top-level help lists deploy/stop/logs/settings/secret."""
        stat, output = self._run_help()
        self.assertEqual(stat, 0, output)
        for sub in ('deploy', 'stop', 'logs', 'settings', 'secret'):
            self.assertIn(sub, output)

    def test_studio_deploy_help(self):
        stat, output = self._run_help('deploy')
        self.assertEqual(stat, 0, output)
        # deploy accepts a repo/studio ID
        self.assertTrue('repo_id' in output or 'studio' in output.lower(),
                        output)

    def test_studio_stop_help(self):
        stat, output = self._run_help('stop')
        self.assertEqual(stat, 0, output)
        self.assertTrue('repo_id' in output or 'studio' in output.lower(),
                        output)

    def test_studio_logs_help(self):
        stat, output = self._run_help('logs')
        self.assertEqual(stat, 0, output)
        self.assertIn('--log-type', output)
        self.assertIn('--keyword', output)
        self.assertIn('--page-num', output)
        self.assertIn('--page-size', output)

    def test_studio_settings_help(self):
        stat, output = self._run_help('settings')
        self.assertEqual(stat, 0, output)
        # settings should accept key=value pairs or specific flags
        self.assertTrue(
            'key=value' in output.lower() or 'settings' in output.lower(),
            output)

    def test_studio_secret_help(self):
        stat, output = self._run_help('secret')
        self.assertEqual(stat, 0, output)
        for sub in ('list', 'add', 'update', 'delete'):
            self.assertIn(sub, output)

    def test_studio_group_lists_secret_and_variable(self):
        """The re-exported ``studio`` group surfaces both leaf families.

        Variables are deliberately separate from secrets: their values are
        public, so the leaf surface must not silently disappear behind the
        re-export.
        """
        stat, output = self._run_help('studio')
        self.assertEqual(stat, 0, output)
        for sub in ('secret', 'variable'):
            self.assertIn(sub, output)

    @unittest.skipUnless(
        _hub_supports_description(),
        'modelscope_hub with secret/variable description support required')
    def test_description_flag_in_help(self):
        """``--description`` is exposed on secret/variable add and update."""
        for cli_args in (('studio', 'secret', 'add'),
                         ('studio', 'secret', 'update'),
                         ('studio', 'variable', 'add'),
                         ('studio', 'variable', 'update')):
            stat, output = self._run_help(*cli_args)
            self.assertEqual(stat, 0, output)
            self.assertIn('--description', output)

    @unittest.skipUnless(
        _hub_supports_studio_oauth(),
        'modelscope_hub with Studio OAuth flags required')
    def test_oauth_flags_in_create_help(self):
        """``create`` exposes the Studio OAuth app configuration flags."""
        stat, output = self._run_help('create')
        self.assertEqual(stat, 0, output)
        for flag in ('--oauth-enabled', '--oauth-scope',
                     '--oauth-token-expire-time', '--oauth-callback-url',
                     '--oauth-homepage-url'):
            self.assertIn(flag, output)

    @unittest.skipUnless(
        _hub_supports_studio_oauth(),
        'modelscope_hub with Studio OAuth flags required')
    def test_oauth_flags_in_settings_help(self):
        """``studio settings`` exposes the Studio OAuth app flags."""
        stat, output = self._run_help('studio', 'settings')
        self.assertEqual(stat, 0, output)
        for flag in ('--oauth-enabled', '--no-oauth-enabled', '--oauth-scope',
                     '--oauth-token-expire-time', '--oauth-callback-url',
                     '--oauth-homepage-url'):
            self.assertIn(flag, output)

    def test_download_repo_type_includes_model(self):
        stat, output = self._run_help('download')
        self.assertEqual(stat, 0, output)
        self.assertIn('model', output)

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
        cmd = StudioCMD(args)
        # The hub leaf reports "no setting specified" as ValueError, which the
        # hub CLI entry point turns into exit code 2.
        with self.assertRaises(ValueError):
            cmd.execute()

    def test_secret_no_action_raises(self):
        args = argparse.Namespace(
            studio_action='secret',
            secret_action=None,
            token=None,
            endpoint=None,
        )
        cmd = StudioCMD(args)
        # Likewise: the hub leaf raises ValueError for an unknown/absent action.
        with self.assertRaises(ValueError):
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
            studios = ', '.join(cls._created_studios)
            logging.getLogger('modelscope').info(
                f'Test studios created (manual cleanup needed): {studios}')

    def _create_and_track(self, **kwargs):
        """Create a studio with unique name, track for cleanup."""
        name = f'ut_test_create_{uuid4().hex[:8]}'
        repo_id = f'{self.owner}/{name}'
        api = HubApi()
        # Merge studio creation defaults from config with caller overrides
        create_kwargs = {
            'sdk_type':
            self.config.get('sdk_type', 'gradio'),
            'sdk_version':
            self.config.get('sdk_version', '6.2.0'),
            'base_image':
            self.config.get('base_image',
                            'ubuntu22.04-py311-torch2.9.1-modelscope1.35.0'),
            'hardware':
            self.config.get('hardware', 'platform/2v-cpu-16g-mem'),
        }
        create_kwargs.update(kwargs)
        print(f'Creating studio {repo_id} with config: {self.config}')
        result = api.create_repo(
            repo_id,
            repo_type='studio',
            visibility=self.config.get('visibility', 'private'),
            license=self.config.get('license', 'apache-2.0'),
            token=self.token,
            endpoint=self.config.get('endpoint'),
            create_default_config=False,
            **create_kwargs,
        )
        self._created_studios.append(repo_id)
        print(f'Created studio {repo_id}: {result}')
        return repo_id, result

    def test_create_studio_basic(self):
        """Create a basic studio and verify it exists."""
        repo_id, result = self._create_and_track(sdk_type='gradio')
        # modelscope_hub's create_repo returns a RepoInfo, not a URL string.
        self.assertEqual(result.id, repo_id)
        # Verify it actually exists
        api = HubApi()
        exists = api.repo_exists(repo_id, repo_type='studio', token=self.token)
        self.assertTrue(exists,
                        f'Studio {repo_id} should exist after creation')

    def test_create_studio_exist_ok(self):
        """Creating an existing studio with exist_ok=True should not raise."""
        repo_id, _ = self._create_and_track(sdk_type='gradio')
        api = HubApi()
        # Create again with exist_ok=True; it must not raise, and the studio
        # must still exist afterwards.
        api.create_repo(
            repo_id,
            repo_type='studio',
            visibility=self.config.get('visibility', 'private'),
            token=self.token,
            endpoint=self.config.get('endpoint'),
            exist_ok=True,
            create_default_config=False,
        )
        self.assertTrue(
            api.repo_exists(repo_id, repo_type='studio', token=self.token))

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


class TestStudioOAuthSettingsDispatch(TestResultMixin, unittest.TestCase):
    """Dispatch-level coverage for the Studio OAuth app configuration.

    Mocked at the ``make_api`` boundary: the OAuth endpoints come from a
    modelscope_service branch that may not be deployed on the configured
    endpoint yet, so the assertions here cover CLI-to-payload plumbing only
    (flag names, tri-state gating, grouping) rather than server round-trips.
    """

    def _settings_namespace(self, **oauth_fields):
        base = {
            'studio_action': 'settings',
            'studio_id': 'owner/demo',
            'settings': [],
            'token': None,
            'endpoint': None,
        }
        base.update(oauth_fields)
        return argparse.Namespace(**base)

    @unittest.skipUnless(
        _hub_supports_studio_oauth(),
        'modelscope_hub with Studio OAuth flags required')
    def test_oauth_flags_reach_update_repo_settings(self):
        from unittest.mock import MagicMock
        args = self._settings_namespace(
            oauth_enabled=True,
            oauth_scope='openid profile',
            oauth_token_expire_time=4,
            oauth_callback_urls=['https://a.example/cb'],
            oauth_homepage_url='https://example.org',
        )
        mock_api = MagicMock()
        with patch('modelscope_hub.cli.studio.make_api',
                   return_value=mock_api):
            StudioCMD(args).execute()
        mock_api.update_repo_settings.assert_called_once_with(
            'owner/demo',
            'studio',
            oauth_enabled=True,
            oauth_scope='openid profile',
            oauth_token_expire_time=4,
            oauth_callback_urls=['https://a.example/cb'],
            oauth_homepage_url='https://example.org',
        )

    @unittest.skipUnless(
        _hub_supports_studio_oauth(),
        'modelscope_hub with Studio OAuth flags required')
    def test_disable_oauth_sends_false(self):
        from unittest.mock import MagicMock
        args = self._settings_namespace(oauth_enabled=False)
        mock_api = MagicMock()
        with patch('modelscope_hub.cli.studio.make_api',
                   return_value=mock_api):
            StudioCMD(args).execute()
        mock_api.update_repo_settings.assert_called_once_with(
            'owner/demo', 'studio', oauth_enabled=False)

    @unittest.skipUnless(
        _hub_supports_studio_oauth(),
        'modelscope_hub with Studio OAuth flags required')
    def test_lone_oauth_scope_without_enabled_is_rejected(self):
        # The server applies the oauth_* group only when oauth_enabled is
        # explicitly present; the CLI must reject a lone scope flag instead of
        # letting the server silently drop it.
        args = self._settings_namespace(oauth_scope='openid')
        with patch('modelscope_hub.cli.studio.make_api') as mock_make:
            with self.assertRaises(ValueError):
                StudioCMD(args).execute()
        mock_make.assert_not_called()


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
        # Track secret/variable keys created during a test for cleanup.
        self._secrets_to_cleanup = []
        self._variables_to_cleanup = []

    def tearDown(self):
        # Best-effort cleanup of any secrets/variables created during the test.
        if not self._cleanup:
            return
        api = HubApi()
        for key in self._secrets_to_cleanup:
            try:
                api.delete_studio_secret(
                    self.studio_id, key, token=self.token, endpoint=None)
            except Exception:
                pass
        for key in self._variables_to_cleanup:
            try:
                api.delete_studio_variable(
                    self.studio_id, key, token=self.token, endpoint=None)
            except Exception:
                pass

    def _unique_secret_key(self):
        """Generate a unique secret key with test prefix."""
        key = f'_TEST_CLI_SECRET_{uuid4().hex[:8].upper()}'
        self._secrets_to_cleanup.append(key)
        return key

    def _unique_variable_key(self):
        """Generate a unique variable key with test prefix."""
        key = f'_TEST_CLI_VARIABLE_{uuid4().hex[:8].upper()}'
        self._variables_to_cleanup.append(key)
        return key

    def _find_secret_description(self, key):
        """Return the stored description for *key* (None when unset/cleared)."""
        api = HubApi()
        for item in api.list_studio_secrets(
                self.studio_id, token=self.token, endpoint=None):
            if isinstance(item, dict) and item.get('key') == key:
                return item.get('description')
        self.fail(f'Secret {key!r} not found in list result')

    def _find_variable(self, key):
        """Return the stored variable dict for *key*."""
        api = HubApi()
        for item in api.list_studio_variables(
                self.studio_id, token=self.token, endpoint=None):
            if isinstance(item, dict) and item.get('key') == key:
                return item
        self.fail(f'Variable {key!r} not found in list result')

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
            log_type='run',
            keyword=None,
            page_num=1,
            page_size=100,
            start_timestamp=None,
            end_timestamp=None,
        )
        # Studio may not be deployed yet, so 404 (no logs) is acceptable.
        from requests.exceptions import HTTPError
        with patch('builtins.print'):
            try:
                StudioCMD(args).execute()
            except HTTPError as e:
                # 404 is expected for a non-deployed studio (no logs available)
                if e.response is not None and e.response.status_code == 404:
                    pass
                else:
                    raise

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
            description=None,
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
            description=None,
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

    @unittest.skipUnless(
        _hub_supports_description(),
        'modelscope_hub with secret/variable description support required')
    def test_secret_add_with_description_direct(self):
        key = self._unique_secret_key()
        args = argparse.Namespace(
            studio_action='secret',
            secret_action='add',
            studio_id=self.studio_id,
            key=key,
            value='described_value',
            description='used by the CLI test suite',
            token=self.token,
            endpoint=None,
        )
        with patch('builtins.print') as mock_print:
            StudioCMD(args).execute()
        printed = ' '.join(
            str(c.args[0]) for c in mock_print.call_args_list if c.args)
        self.assertIn('added', printed.lower())
        self.assertEqual(
            self._find_secret_description(key), 'used by the CLI test suite')

    @unittest.skipUnless(
        _hub_supports_description(),
        'modelscope_hub with secret/variable description support required')
    def test_secret_update_description_direct(self):
        key = self._unique_secret_key()
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
            description='replacement description',
            token=self.token,
            endpoint=None,
        )
        with patch('builtins.print') as mock_print:
            StudioCMD(args).execute()
        printed = ' '.join(
            str(c.args[0]) for c in mock_print.call_args_list if c.args)
        self.assertIn('updated', printed.lower())
        self.assertEqual(
            self._find_secret_description(key), 'replacement description')

        # Omitting description on a further update keeps the stored one
        # (server treats the absent field as a pointer that must not rewrite
        # the column; only an explicit empty string clears it).
        args = argparse.Namespace(
            studio_action='secret',
            secret_action='update',
            studio_id=self.studio_id,
            key=key,
            value='updated_again',
            description=None,
            token=self.token,
            endpoint=None,
        )
        with patch('builtins.print'):
            StudioCMD(args).execute()
        self.assertEqual(
            self._find_secret_description(key), 'replacement description')

    @unittest.skipUnless(
        _hub_supports_description(),
        'modelscope_hub with secret/variable description support required')
    def test_secret_update_description_clear_direct(self):
        """Passing an empty description clears the stored one; it must not be
        conflated with omitting the field (which keeps the stored value)."""
        key = self._unique_secret_key()
        args = argparse.Namespace(
            studio_action='secret',
            secret_action='add',
            studio_id=self.studio_id,
            key=key,
            value='described_value',
            description='will be cleared',
            token=self.token,
            endpoint=None,
        )
        with patch('builtins.print'):
            StudioCMD(args).execute()
        self.assertEqual(self._find_secret_description(key), 'will be cleared')

        args = argparse.Namespace(
            studio_action='secret',
            secret_action='update',
            studio_id=self.studio_id,
            key=key,
            value='updated_value',
            description='',
            token=self.token,
            endpoint=None,
        )
        with patch('builtins.print'):
            StudioCMD(args).execute()
        self.assertFalse(self._find_secret_description(key))

    def test_variable_add_direct(self):
        key = self._unique_variable_key()
        args = argparse.Namespace(
            studio_action='variable',
            variable_action='add',
            studio_id=self.studio_id,
            key=key,
            value='var_value',
            description=None,
            token=self.token,
            endpoint=None,
        )
        with patch('builtins.print') as mock_print:
            StudioCMD(args).execute()
        printed = ' '.join(
            str(c.args[0]) for c in mock_print.call_args_list if c.args)
        self.assertIn('added', printed.lower())

    def test_variable_update_direct(self):
        key = self._unique_variable_key()
        api = HubApi()
        api.add_studio_variable(
            self.studio_id,
            key,
            'initial_value',
            token=self.token,
            endpoint=None)

        args = argparse.Namespace(
            studio_action='variable',
            variable_action='update',
            studio_id=self.studio_id,
            key=key,
            value='updated_value',
            description=None,
            token=self.token,
            endpoint=None,
        )
        with patch('builtins.print') as mock_print:
            StudioCMD(args).execute()
        printed = ' '.join(
            str(c.args[0]) for c in mock_print.call_args_list if c.args)
        self.assertIn('updated', printed.lower())

    def test_variable_list_direct(self):
        args = argparse.Namespace(
            studio_action='variable',
            variable_action='list',
            studio_id=self.studio_id,
            token=self.token,
            endpoint=None,
        )
        # Should not raise regardless of whether variables exist.
        with patch('builtins.print'):
            StudioCMD(args).execute()

    @unittest.skipUnless(
        _hub_supports_description(),
        'modelscope_hub with secret/variable description support required')
    def test_variable_add_with_description_direct(self):
        key = self._unique_variable_key()
        args = argparse.Namespace(
            studio_action='variable',
            variable_action='add',
            studio_id=self.studio_id,
            key=key,
            value='public_value',
            description='plaintext doc',
            token=self.token,
            endpoint=None,
        )
        with patch('builtins.print') as mock_print:
            StudioCMD(args).execute()
        printed = ' '.join(
            str(c.args[0]) for c in mock_print.call_args_list if c.args)
        self.assertIn('added', printed.lower())
        item = self._find_variable(key)
        self.assertEqual(item.get('value'), 'public_value')
        self.assertEqual(item.get('description'), 'plaintext doc')

    def test_variable_delete_direct(self):
        key = self._unique_variable_key()
        api = HubApi()
        api.add_studio_variable(
            self.studio_id,
            key,
            'to_be_deleted',
            token=self.token,
            endpoint=None)
        # Remove from cleanup list since we expect CLI to delete it.
        self._variables_to_cleanup.remove(key)

        args = argparse.Namespace(
            studio_action='variable',
            variable_action='delete',
            studio_id=self.studio_id,
            key=key,
            yes=True,
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
