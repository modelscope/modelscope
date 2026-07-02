# Copyright (c) ModelScope Contributors. All rights reserved.
"""Smoke tests for `modelscope agent` CLI — real environment, real server.

Covers every subcommand and parameter with actual token and endpoint.
Uses the user's logged-in modelscope session (or TOKEN / SERVER env vars).

Usage:
    TOKEN=xxx python -m pytest tests/cli/test_agent_cmd.py -v
    # Or if already logged in via `modelscope login`:
    python -m pytest tests/cli/test_agent_cmd.py -v
"""
import io
import os
import sys
import tempfile
import time
import unittest
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

from modelscope.cli.agent import AgentCMD, _get_config, _normalize_endpoint

# ---------------------------------------------------------------------------
# Config — real environment
# ---------------------------------------------------------------------------
SERVER = os.environ.get("SERVER", "http://pre.modelscope.cn").strip()
TOKEN = os.environ.get("TOKEN", "").strip() or None    # None = auto from HubConfig
# Public repo for anonymous download test (owner/name format)
PUBLIC_REPO = os.environ.get("PUBLIC_REPO", "tastelikefeet/test-agent-integration")
# Framework to test with
TEST_FRAMEWORK = os.environ.get("TEST_FRAMEWORK", "nanobot")
# Request interval to avoid 429
REQUEST_INTERVAL = int(os.environ.get("REQUEST_INTERVAL", "2"))


def _get_real_config():
    """Get real HubConfig from environment/login session."""
    from modelscope_hub.config import HubConfig
    config = HubConfig(endpoint=SERVER, token=TOKEN)
    if config.endpoint:
        config.endpoint = _normalize_endpoint(config.endpoint)
    return config


def _real_token():
    """Return real token (from env or login session)."""
    return _get_real_config().token


def _real_endpoint():
    """Return real endpoint."""
    cfg = _get_real_config()
    return cfg.endpoint or 'https://modelscope.cn'


def _throttle():
    time.sleep(REQUEST_INTERVAL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_parser():
    """Build argparser with AgentCMD registered."""
    p = ArgumentParser()
    sub = p.add_subparsers()
    AgentCMD.register(sub)
    return p


def _parse(argv):
    """Parse argv and return namespace."""
    return _build_parser().parse_args(['agent'] + argv)


def _make_cmd(argv):
    """Parse and instantiate AgentCMD ready for execute()."""
    args = _parse(argv)
    cmd = object.__new__(AgentCMD)
    cmd.args = args
    # Inject real endpoint/token into args (simulating global CLI flags)
    if not hasattr(args, 'endpoint') or not args.endpoint:
        args.endpoint = _real_endpoint()
    if not hasattr(args, 'token') or not args.token:
        args.token = _real_token()
    return cmd


def _capture(fn):
    """Run fn() capturing stdout+stderr, return (stdout_str, stderr_str, exception_or_None)."""
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = out_buf
    sys.stderr = err_buf
    exc = None
    try:
        fn()
    except SystemExit as e:
        exc = e
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
    return out_buf.getvalue(), err_buf.getvalue(), exc


# ===========================================================================
# 1. Argument Parsing — pure logic, no network
# ===========================================================================

class TestArgParsing(unittest.TestCase):
    """Verify argparse correctly maps all flags to args namespace."""

    # ---- upload ----
    def test_upload_all_params(self):
        args = _parse([
            'upload', '-f', 'qoder', '-n', 'reviewer',
            '-r', 'org/repo', '--local-dir', '/tmp/ws', '--dry-run',
        ])
        self.assertEqual(args.agent_action, 'upload')
        self.assertEqual(args.framework, 'qoder')
        self.assertEqual(args.name, 'reviewer')
        self.assertEqual(args.repo, 'org/repo')
        self.assertEqual(args.local_dir, '/tmp/ws')
        self.assertTrue(args.dry_run)

    def test_upload_minimal(self):
        args = _parse(['upload', '-f', 'nanobot', '-r', 'my-repo'])
        self.assertEqual(args.framework, 'nanobot')
        self.assertEqual(args.repo, 'my-repo')
        self.assertIsNone(args.name)
        self.assertFalse(args.dry_run)

    def test_upload_missing_framework_exits(self):
        with self.assertRaises(SystemExit):
            _parse(['upload', '-r', 'repo'])

    def test_upload_missing_repo_exits(self):
        with self.assertRaises(SystemExit):
            _parse(['upload', '-f', 'qoder'])

    # ---- download ----
    def test_download_all_params(self):
        args = _parse([
            'download', '-f', 'nanobot', '-r', 'alice/bot',
            '-n', 'myagent', '--local-dir', '/tmp/out',
            '--target-framework', 'hermes', '--dry-run',
        ])
        self.assertEqual(args.agent_action, 'download')
        self.assertEqual(args.framework, 'nanobot')
        self.assertEqual(args.repo, 'alice/bot')
        self.assertEqual(args.name, 'myagent')
        self.assertEqual(args.target_framework, 'hermes')
        self.assertTrue(args.dry_run)

    def test_download_minimal(self):
        args = _parse(['download', '-f', 'hermes', '-r', 'x/y'])
        self.assertEqual(args.framework, 'hermes')
        self.assertIsNone(args.target_framework)
        self.assertFalse(args.dry_run)

    def test_download_missing_repo_exits(self):
        with self.assertRaises(SystemExit):
            _parse(['download', '-f', 'nanobot'])

    # ---- watch ----
    def test_watch_all_params(self):
        args = _parse([
            'watch', '-f', 'qoder', '-n', 'reviewer',
            '-r', 'org/repo', '--local-dir', '/ws', '--pull',
        ])
        self.assertEqual(args.agent_action, 'watch')
        self.assertEqual(args.framework, 'qoder')
        self.assertEqual(args.name, 'reviewer')
        self.assertTrue(args.pull)

    def test_watch_minimal(self):
        args = _parse(['watch', '-f', 'nanobot', '-r', 'repo'])
        self.assertFalse(args.pull)
        self.assertIsNone(args.name)

    def test_watch_missing_framework_exits(self):
        with self.assertRaises(SystemExit):
            _parse(['watch', '-r', 'repo'])

    # ---- list ----
    def test_list_all_params(self):
        args = _parse(['list', '--owner', 'alice', '--page', '3', '--page-size', '20'])
        self.assertEqual(args.owner, 'alice')
        self.assertEqual(args.page_number, 3)
        self.assertEqual(args.page_size, 20)

    def test_list_defaults(self):
        args = _parse(['list'])
        self.assertIsNone(args.owner)
        self.assertEqual(args.page_number, 1)
        self.assertEqual(args.page_size, 10)

    # ---- status ----
    def test_status_all_params(self):
        args = _parse(['status', '-f', 'qoder', '--local-dir', '/ws'])
        self.assertEqual(args.framework, 'qoder')
        self.assertEqual(args.local_dir, '/ws')

    def test_status_missing_framework_exits(self):
        with self.assertRaises(SystemExit):
            _parse(['status'])

    # ---- backups ----
    def test_backups_all_params(self):
        args = _parse(['backups', '-f', 'nanobot', '-n', 'bot', '--local-dir', '/d'])
        self.assertEqual(args.framework, 'nanobot')
        self.assertEqual(args.name, 'bot')

    def test_backups_defaults(self):
        args = _parse(['backups'])
        self.assertIsNone(args.framework)
        self.assertIsNone(args.name)

    # ---- restore ----
    def test_restore_all_params(self):
        args = _parse([
            'restore', '--from-backup', 'last',
            '-f', 'qoder', '-n', 'reviewer', '--local-dir', '/d',
        ])
        self.assertEqual(args.from_backup, 'last')
        self.assertEqual(args.framework, 'qoder')

    def test_restore_missing_from_backup_exits(self):
        with self.assertRaises(SystemExit):
            _parse(['restore', '-f', 'qoder'])

    # ---- convert ----
    def test_convert_all_params(self):
        args = _parse([
            'convert', '--from-framework', 'nanobot',
            '--target-framework', 'hermes',
            '--from-name', 'reviewer', '--target-name', 'assistant',
            '--local-dir', '/src', '--out', '/dst', '--dry-run',
        ])
        self.assertEqual(args.from_framework, 'nanobot')
        self.assertEqual(args.target_framework, 'hermes')
        self.assertEqual(args.from_name, 'reviewer')
        self.assertEqual(args.target_name, 'assistant')
        self.assertTrue(args.dry_run)

    def test_convert_minimal(self):
        args = _parse([
            'convert', '--from-framework', 'nanobot',
            '--target-framework', 'qoder',
        ])
        self.assertIsNone(args.from_name)
        self.assertIsNone(args.target_name)
        self.assertFalse(args.dry_run)

    def test_convert_missing_from_framework_exits(self):
        with self.assertRaises(SystemExit):
            _parse(['convert', '--target-framework', 'hermes'])

    def test_convert_missing_target_framework_exits(self):
        with self.assertRaises(SystemExit):
            _parse(['convert', '--from-framework', 'nanobot'])

    # ---- stop ----
    def test_stop_parses(self):
        args = _parse(['stop'])
        self.assertEqual(args.agent_action, 'stop')


# ===========================================================================
# 2. Real Execution — list
# ===========================================================================

class TestListReal(unittest.TestCase):
    """List remote agent repos — hits real server.

    If GET /agents is not deployed, tests verify error handling.
    If deployed, tests verify table output.
    """

    def _run_list(self, extra_args=None):
        cmd = _make_cmd(['list'] + (extra_args or []))
        return _capture(cmd.execute)

    def test_list_default(self):
        out, err, exc = self._run_list()
        if exc is None:
            # Endpoint available — should print table or empty message
            self.assertTrue(
                'repo_id' in out or 'no agent repositories found' in out,
                f"unexpected output: {out}")
        else:
            # Endpoint not deployed — should exit with clear error
            self.assertIn('404', err)

    def test_list_with_page_size(self):
        _throttle()
        out, err, exc = self._run_list(['--page-size', '3'])
        if exc is None:
            self.assertIn('page_size=3', out)
        else:
            self.assertIn('list failed', err)

    def test_list_with_owner(self):
        _throttle()
        owner = PUBLIC_REPO.split('/')[0] if '/' in PUBLIC_REPO else 'tastelikefeet'
        out, err, exc = self._run_list(['--owner', owner, '--page-size', '5'])
        if exc is None:
            self.assertTrue(len(out) > 0)
        else:
            self.assertIn('list failed', err)

    def test_list_page_2(self):
        _throttle()
        out, err, exc = self._run_list(['--page', '2', '--page-size', '5'])
        if exc is None:
            self.assertIn('page 2', out)
        else:
            self.assertIn('list failed', err)


# ===========================================================================
# 3. Real Execution — download (--dry-run, public repo, no token needed)
# ===========================================================================

class TestDownloadReal(unittest.TestCase):
    """Download from real server — dry-run to avoid writing files."""

    def test_download_dry_run_public_repo(self):
        _throttle()
        tmp = tempfile.mkdtemp()
        cmd = _make_cmd([
            'download', '-f', TEST_FRAMEWORK, '-r', PUBLIC_REPO,
            '--local-dir', tmp, '--dry-run',
        ])
        out, err, exc = _capture(cmd.execute)
        self.assertIsNone(exc, f"download failed: {err}")
        self.assertIn('dry-run', out)

    def test_download_with_target_framework_dry_run(self):
        _throttle()
        tmp = tempfile.mkdtemp()
        cmd = _make_cmd([
            'download', '-f', TEST_FRAMEWORK, '-r', PUBLIC_REPO,
            '--target-framework', 'hermes',
            '--local-dir', tmp, '--dry-run',
        ])
        out, err, exc = _capture(cmd.execute)
        self.assertIsNone(exc, f"download failed: {err}")
        # Should mention conversion
        self.assertTrue('Converted' in out or 'dry-run' in out,
                        f"unexpected: {out}")

    def test_download_with_name_param(self):
        _throttle()
        tmp = tempfile.mkdtemp()
        cmd = _make_cmd([
            'download', '-f', TEST_FRAMEWORK, '-r', PUBLIC_REPO,
            '-n', 'myagent', '--local-dir', tmp, '--dry-run',
        ])
        out, err, exc = _capture(cmd.execute)
        self.assertIsNone(exc, f"download failed: {err}")
        self.assertIn('dry-run', out)

    def test_download_unknown_framework_exits(self):
        cmd = _make_cmd([
            'download', '-f', 'nonexist_fw_xyz', '-r', PUBLIC_REPO,
        ])
        _, err, exc = _capture(cmd.execute)
        self.assertIsNotNone(exc)

    def test_download_nonexistent_repo_exits(self):
        _throttle()
        tmp = tempfile.mkdtemp()
        cmd = _make_cmd([
            'download', '-f', TEST_FRAMEWORK,
            '-r', 'nonexist_user_xyz/nonexist_repo_xyz',
            '--local-dir', tmp, '--dry-run',
        ])
        _, err, exc = _capture(cmd.execute)
        self.assertIsNotNone(exc)


# ===========================================================================
# 4. Real Execution — upload (--dry-run, needs token)
# ===========================================================================

class TestUploadReal(unittest.TestCase):
    """Upload dry-run — needs real token."""

    def setUp(self):
        if not _real_token():
            self.skipTest("no token available (set TOKEN env or modelscope login)")
        self.tmp = tempfile.mkdtemp()
        p = Path(self.tmp)
        (p / 'agents').mkdir()
        (p / 'agents' / 'reviewer.md').write_text('# Reviewer\nTest agent.')
        (p / 'AGENTS.md').write_text('# Shared prompt')

    def test_upload_dry_run(self):
        _throttle()
        cmd = _make_cmd([
            'upload', '-f', 'qoder', '-n', 'reviewer',
            '-r', PUBLIC_REPO, '--local-dir', self.tmp, '--dry-run',
        ])
        out, err, exc = _capture(cmd.execute)
        self.assertIsNone(exc, f"upload dry-run failed: {err}")
        self.assertIn('dry-run', out)
        self.assertIn('reviewer.md', out)

    def test_upload_unknown_framework_exits(self):
        cmd = _make_cmd([
            'upload', '-f', 'nonexist_fw', '-n', 'x',
            '-r', 'org/repo', '--local-dir', self.tmp,
        ])
        _, err, exc = _capture(cmd.execute)
        self.assertIsNotNone(exc)

    def test_upload_no_files_exits(self):
        _throttle()
        empty = tempfile.mkdtemp()
        cmd = _make_cmd([
            'upload', '-f', 'nanobot', '-n', 'default',
            '-r', PUBLIC_REPO, '--local-dir', empty,
        ])
        _, err, exc = _capture(cmd.execute)
        self.assertIsNotNone(exc)


# ===========================================================================
# 5. Real Execution — watch (validation only, don't actually start daemon)
# ===========================================================================

class TestWatchReal(unittest.TestCase):
    """Watch validation — tests that error early on bad input."""

    def test_watch_unknown_framework_exits(self):
        cmd = _make_cmd(['watch', '-f', 'nonexist_fw', '-r', 'repo'])
        _, err, exc = _capture(cmd.execute)
        self.assertIsNotNone(exc)


# ===========================================================================
# 6. Real Execution — convert (local only, no network)
# ===========================================================================

class TestConvertReal(unittest.TestCase):
    """Convert — local operation, uses real allowlist registry."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        p = Path(self.tmp)
        (p / 'SOUL.md').write_text('# Soul\nI am a nanobot agent.')
        (p / 'USER.md').write_text('# User\nPreferences here.')
        (p / 'memory').mkdir()
        (p / 'memory' / 'MEMORY.md').write_text('# Memory\nPast events.')

    def test_convert_dry_run(self):
        out_dir = tempfile.mkdtemp()
        cmd = _make_cmd([
            'convert', '--from-framework', 'nanobot',
            '--target-framework', 'hermes',
            '--from-name', 'default', '--target-name', 'assistant',
            '--local-dir', self.tmp, '--out', out_dir, '--dry-run',
        ])
        out, err, exc = _capture(cmd.execute)
        self.assertIsNone(exc, f"convert dry-run failed: {err}")
        self.assertIn('dry-run', out)
        self.assertIn('nanobot/default', out)
        self.assertIn('hermes/assistant', out)

    def test_convert_actually_writes(self):
        out_dir = tempfile.mkdtemp()
        cmd = _make_cmd([
            'convert', '--from-framework', 'nanobot',
            '--target-framework', 'hermes',
            '--local-dir', self.tmp, '--out', out_dir,
        ])
        out, err, exc = _capture(cmd.execute)
        self.assertIsNone(exc, f"convert write failed: {err}")
        self.assertIn('Wrote', out)
        # hermes layout: USER.md -> memories/USER.md
        out_path = Path(out_dir)
        self.assertTrue(
            (out_path / 'memories' / 'USER.md').exists() or
            (out_path / 'SOUL.md').exists(),
            f"no output files in {out_dir}")

    def test_convert_target_name_defaults_to_from_name(self):
        out_dir = tempfile.mkdtemp()
        cmd = _make_cmd([
            'convert', '--from-framework', 'nanobot',
            '--target-framework', 'hermes',
            '--from-name', 'mybot',
            '--local-dir', self.tmp, '--out', out_dir, '--dry-run',
        ])
        out, err, exc = _capture(cmd.execute)
        self.assertIsNone(exc)
        self.assertIn('hermes/mybot', out)

    def test_convert_unknown_source_framework_exits(self):
        cmd = _make_cmd([
            'convert', '--from-framework', 'nonexist',
            '--target-framework', 'hermes', '--local-dir', self.tmp,
        ])
        _, _, exc = _capture(cmd.execute)
        self.assertIsNotNone(exc)

    def test_convert_unknown_target_framework_exits(self):
        cmd = _make_cmd([
            'convert', '--from-framework', 'nanobot',
            '--target-framework', 'nonexist', '--local-dir', self.tmp,
        ])
        _, _, exc = _capture(cmd.execute)
        self.assertIsNotNone(exc)

    def test_convert_no_source_files_exits(self):
        empty = tempfile.mkdtemp()
        cmd = _make_cmd([
            'convert', '--from-framework', 'nanobot',
            '--target-framework', 'hermes', '--local-dir', empty,
        ])
        _, _, exc = _capture(cmd.execute)
        self.assertIsNotNone(exc)


# ===========================================================================
# 7. Real Execution — status / backups / restore / stop (local only)
# ===========================================================================

class TestStatusReal(unittest.TestCase):
    """Status — local only, shows framework agent list."""

    def test_status_valid_framework(self):
        tmp = tempfile.mkdtemp()
        cmd = _make_cmd(['status', '-f', 'qoder', '--local-dir', tmp])
        out, err, exc = _capture(cmd.execute)
        self.assertIsNone(exc, f"status failed: {err}")

    def test_status_unknown_framework_exits(self):
        cmd = _make_cmd(['status', '-f', 'nonexist_fw'])
        _, _, exc = _capture(cmd.execute)
        self.assertIsNotNone(exc)


class TestBackupsReal(unittest.TestCase):
    """Backups — local only."""

    def test_backups_empty_dir(self):
        tmp = tempfile.mkdtemp()
        cmd = _make_cmd(['backups', '-f', 'qoder', '--local-dir', tmp])
        out, err, exc = _capture(cmd.execute)
        # May return 0 with "no backups" or just succeed
        self.assertTrue(exc is None or (exc and exc.code == 0),
                        f"backups failed: {err}")


class TestRestoreReal(unittest.TestCase):
    """Restore — local only."""

    def test_restore_no_backup_exits(self):
        tmp = tempfile.mkdtemp()
        cmd = _make_cmd([
            'restore', '--from-backup', 'last',
            '-f', 'qoder', '-n', 'reviewer', '--local-dir', tmp,
        ])
        _, _, exc = _capture(cmd.execute)
        # Should exit because no backups exist
        self.assertIsNotNone(exc)


class TestStopReal(unittest.TestCase):
    """Stop — kills watch process if running."""

    def test_stop_no_running_process(self):
        cmd = _make_cmd(['stop'])
        out, err, exc = _capture(cmd.execute)
        self.assertIsNone(exc, f"stop failed: {err}")
        self.assertIn('No watch process running', out)


# ===========================================================================
# 8. No-action prints usage
# ===========================================================================

class TestNoAction(unittest.TestCase):
    def test_no_subcommand_prints_usage(self):
        cmd = _make_cmd([])
        out, err, exc = _capture(cmd.execute)
        self.assertIsNone(exc)
        self.assertIn('Usage:', out)


if __name__ == "__main__":
    unittest.main()
