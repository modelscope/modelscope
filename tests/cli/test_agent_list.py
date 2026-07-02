# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for agent CLI: _resolve_repo, _normalize_endpoint, _get_username, list_agents, download.

Uses REAL environment — real token, real server, no mocks.
Token resolved from login session or TOKEN env var.

Usage:
    TOKEN=xxx python -m pytest tests/cli/test_agent_list.py -v
    # Or if already logged in via `modelscope login`:
    python -m pytest tests/cli/test_agent_list.py -v
"""
import io
import os
import sys
import tempfile
import time
import unittest

from modelscope.cli.agent import (
    AgentCMD,
    _get_config,
    _get_username,
    _normalize_endpoint,
    _resolve_repo,
)
from ultron.cli.client import ApiError, UltronClient

# ---------------------------------------------------------------------------
# Config — real environment
# ---------------------------------------------------------------------------
SERVER = os.environ.get("SERVER", "http://pre.modelscope.cn").strip()
TOKEN = os.environ.get("TOKEN", "").strip() or None
PUBLIC_REPO = os.environ.get("PUBLIC_REPO", "tastelikefeet/test-agent-integration")
REQUEST_INTERVAL = int(os.environ.get("REQUEST_INTERVAL", "2"))


def _get_real_config():
    """Get real HubConfig from environment/login session."""
    from modelscope_hub.config import HubConfig
    config = HubConfig(endpoint=SERVER, token=TOKEN)
    if config.endpoint:
        config.endpoint = _normalize_endpoint(config.endpoint)
    return config


def _real_token():
    return _get_real_config().token


def _real_endpoint():
    cfg = _get_real_config()
    return cfg.endpoint or 'https://modelscope.cn'


def _throttle():
    time.sleep(REQUEST_INTERVAL)


# ---------------------------------------------------------------------------
# 1. Unit tests for _resolve_repo helper (pure logic)
# ---------------------------------------------------------------------------

class TestResolveRepo(unittest.TestCase):
    """Test the _resolve_repo utility function from agent.py."""

    def test_slash_format_splits_correctly(self):
        self.assertEqual(_resolve_repo('alice/my-agent'), ('alice', 'my-agent'))

    def test_slash_format_ignores_username(self):
        self.assertEqual(_resolve_repo('org/repo', username='bob'), ('org', 'repo'))

    def test_slash_with_multiple_slashes(self):
        self.assertEqual(_resolve_repo('org/sub/path'), ('org', 'sub/path'))

    def test_no_slash_uses_username(self):
        self.assertEqual(_resolve_repo('my-repo', 'alice'), ('alice', 'my-repo'))

    def test_no_slash_empty_username(self):
        self.assertEqual(_resolve_repo('my-repo', ''), ('', 'my-repo'))


# ---------------------------------------------------------------------------
# 2. Unit tests for _normalize_endpoint (pure logic)
# ---------------------------------------------------------------------------

class TestNormalizeEndpoint(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(_normalize_endpoint(''), '')

    def test_adds_https_scheme(self):
        self.assertEqual(_normalize_endpoint('modelscope.cn'), 'https://modelscope.cn')

    def test_preserves_http(self):
        self.assertEqual(_normalize_endpoint('http://localhost:8080'), 'http://localhost:8080')

    def test_preserves_https(self):
        self.assertEqual(_normalize_endpoint('https://api.example.com'), 'https://api.example.com')

    def test_strips_trailing_slash(self):
        self.assertEqual(_normalize_endpoint('https://api.example.com/'), 'https://api.example.com')

    def test_adds_scheme_and_strips_slash(self):
        self.assertEqual(_normalize_endpoint('modelscope.cn/'), 'https://modelscope.cn')


# ---------------------------------------------------------------------------
# 3. _get_username — real server call (needs token)
# ---------------------------------------------------------------------------

class TestGetUsername(unittest.TestCase):
    """Test _get_username with real server — requires login."""

    def setUp(self):
        if not _real_token():
            self.skipTest("no token available")

    def test_returns_nonempty_username(self):
        config = _get_real_config()
        username = _get_username(config)
        self.assertIsInstance(username, str)
        self.assertTrue(len(username) > 0, "username should not be empty")
        print(f"  resolved username: {username}")


# ---------------------------------------------------------------------------
# 4. UltronClient.list_agents — real server call
# ---------------------------------------------------------------------------

class TestListAgentsClient(unittest.TestCase):
    """Test list_agents method against real server.

    If GET /agents returns 404, tests verify ApiError is raised correctly.
    """

    def _make_client(self):
        config = _get_real_config()
        return UltronClient(config.endpoint, config.token or '')

    def test_list_agents_default(self):
        client = self._make_client()
        try:
            result = client.list_agents()
            self.assertIn('items', result)
            self.assertIn('total_count', result)
            self.assertIsInstance(result['items'], list)
            self.assertIsInstance(result['total_count'], int)
            print(f"  total_count={result['total_count']}, items={len(result['items'])}")
        except ApiError as e:
            # Endpoint not deployed — verify error is properly raised
            self.assertEqual(e.status, 404)
            print(f"  GET /agents returned {e.status} (not deployed)")

    def test_list_agents_with_owner(self):
        _throttle()
        owner = PUBLIC_REPO.split('/')[0] if '/' in PUBLIC_REPO else 'tastelikefeet'
        client = self._make_client()
        try:
            result = client.list_agents(owner=owner, page_size=5)
            self.assertIn('items', result)
            self.assertIsInstance(result['total_count'], int)
            print(f"  owner={owner}, total_count={result['total_count']}")
        except ApiError as e:
            self.assertEqual(e.status, 404)

    def test_list_agents_pagination(self):
        _throttle()
        client = self._make_client()
        try:
            result = client.list_agents(page_number=1, page_size=3)
            self.assertIn('items', result)
            self.assertLessEqual(len(result['items']), 3)
        except ApiError as e:
            self.assertEqual(e.status, 404)

    def test_list_agents_page_2(self):
        _throttle()
        client = self._make_client()
        try:
            result = client.list_agents(page_number=2, page_size=5)
            self.assertIn('items', result)
        except ApiError as e:
            self.assertEqual(e.status, 404)


# ---------------------------------------------------------------------------
# 5. Agent CLI _list() rendering — real server
# ---------------------------------------------------------------------------

class TestAgentListCLI(unittest.TestCase):
    """Test the _list() subcommand with real server output."""

    def _run_list(self, extra_args=None):
        from argparse import ArgumentParser
        p = ArgumentParser()
        sub = p.add_subparsers()
        AgentCMD.register(sub)
        argv = ['agent', 'list'] + (extra_args or [])
        args = p.parse_args(argv)
        # Inject real endpoint/token
        args.endpoint = _real_endpoint()
        args.token = _real_token()

        cmd = object.__new__(AgentCMD)
        cmd.args = args

        out_buf = io.StringIO()
        err_buf = io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = out_buf
        sys.stderr = err_buf
        exc = None
        try:
            cmd.execute()
        except SystemExit as e:
            exc = e
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
        return out_buf.getvalue(), err_buf.getvalue(), exc

    def test_list_renders_table_or_empty(self):
        out, err, exc = self._run_list()
        if exc is None:
            self.assertTrue(
                'repo_id' in out or 'no agent repositories found' in out,
                f"unexpected output: {out[:200]}")
        else:
            # 404 = endpoint not deployed; verify error message is clear
            self.assertIn('list failed', err)

    def test_list_with_page_size(self):
        _throttle()
        out, err, exc = self._run_list(['--page-size', '2'])
        if exc is None:
            self.assertIn('page_size=2', out)
        else:
            self.assertIn('list failed', err)

    def test_list_with_owner_filter(self):
        _throttle()
        owner = PUBLIC_REPO.split('/')[0]
        out, err, exc = self._run_list(['--owner', owner])
        if exc is None:
            self.assertTrue(len(out) > 0)
        else:
            self.assertIn('list failed', err)


# ---------------------------------------------------------------------------
# 6. Anonymous download — real server, public repo, no token
# ---------------------------------------------------------------------------

class TestAnonymousDownload(unittest.TestCase):
    """Download from public repo without token — real server."""

    def test_anonymous_download_dry_run(self):
        _throttle()
        from argparse import ArgumentParser
        p = ArgumentParser()
        sub = p.add_subparsers()
        AgentCMD.register(sub)

        tmp = tempfile.mkdtemp()
        argv = ['agent', 'download', '-f', 'nanobot', '-r', PUBLIC_REPO,
                '--local-dir', tmp, '--dry-run']
        args = p.parse_args(argv)
        # Force no token — anonymous access
        args.endpoint = _real_endpoint()
        args.token = None

        cmd = object.__new__(AgentCMD)
        cmd.args = args

        buf = io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = buf
        sys.stderr = io.StringIO()
        exc = None
        try:
            cmd.execute()
        except SystemExit as e:
            exc = e
        finally:
            sys.stdout = old_out
            sys.stderr = old_err

        out = buf.getvalue()
        self.assertIsNone(exc, f"anonymous download failed: {out}")
        self.assertIn('dry-run', out)

    def test_anonymous_download_no_slash_repo_fails(self):
        """--repo 'bot' without token should fail (needs username)."""
        from argparse import ArgumentParser
        p = ArgumentParser()
        sub = p.add_subparsers()
        AgentCMD.register(sub)

        argv = ['agent', 'download', '-f', 'nanobot', '-r', 'my-repo-no-slash',
                '--local-dir', tempfile.mkdtemp(), '--dry-run']
        args = p.parse_args(argv)
        args.endpoint = _real_endpoint()
        args.token = None

        cmd = object.__new__(AgentCMD)
        cmd.args = args

        old_err = sys.stderr
        sys.stderr = io.StringIO()
        try:
            with self.assertRaises(SystemExit):
                cmd.execute()
        finally:
            sys.stderr = old_err


if __name__ == "__main__":
    unittest.main()
