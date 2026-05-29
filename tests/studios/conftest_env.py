# Copyright (c) Alibaba, Inc. and its affiliates.
"""Helper module for loading Studio test environment from a local .env file.

The ``.env`` file (sibling of this module) is optional and not committed to
version control. ``.env.example`` documents the supported variables.
"""
import os


def load_test_env():
    """Load test environment variables from a local ``.env`` file if present.

    Existing environment variables take precedence (``setdefault`` semantics)
    so that CI overrides remain authoritative.
    """
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if not os.path.exists(env_file):
        return
    with open(env_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip())


def get_test_config():
    """Return the merged Studio test configuration dictionary."""
    load_test_env()
    cleanup_raw = os.environ.get('TEST_CLEANUP_REMOTE', 'true').strip().lower()
    return {
        'token': os.environ.get('MODELSCOPE_API_TOKEN'),
        'owner': os.environ.get('TEST_STUDIO_OWNER', 'test_user'),
        'visibility': os.environ.get('TEST_STUDIO_VISIBILITY', 'private'),
        'endpoint': os.environ.get('MODELSCOPE_ENDPOINT',
                                   'https://modelscope.cn'),
        'studio_id': os.environ.get('TEST_STUDIO_ID'),
        'cleanup': cleanup_raw not in ('false', '0', 'no'),
    }


def create_temp_studio(config, name_prefix='_test_cli'):
    """Create a temporary studio for testing purposes.

    Args:
        config: dict from get_test_config()
        name_prefix: prefix for the generated studio name

    Returns:
        str: The studio_id (owner/name) of the created studio.

    Raises:
        unittest.SkipTest: If token or owner are not configured.
    """
    import unittest
    from uuid import uuid4
    from modelscope.hub.api import HubApi

    token = config['token']
    owner = config['owner']
    if not token or not owner:
        raise unittest.SkipTest(
            'MODELSCOPE_API_TOKEN and TEST_STUDIO_OWNER required')

    name = f'{name_prefix}_{uuid4().hex[:8]}'
    repo_id = f'{owner}/{name}'
    visibility = config.get('visibility', 'private')

    api = HubApi()
    api.create_repo(
        repo_id,
        repo_type='studio',
        visibility=visibility,
        token=token,
        endpoint=config.get('endpoint'),
        sdk_type='gradio',
        create_default_config=False,
    )
    return repo_id


class TestResultMixin:
    """Mixin that prints test pass/fail status after each test method.

    Must be placed BEFORE ``unittest.TestCase`` in the MRO so that this
    ``tearDown`` runs and still chains to ``TestCase.tearDown`` via ``super``.
    """

    def tearDown(self):
        try:
            super().tearDown()
        finally:
            status = 'PASSED'
            outcome = getattr(self, '_outcome', None)
            if outcome is not None:
                # Python <= 3.10 exposes ``result`` directly.
                result = getattr(outcome, 'result', None)
                if result is not None and hasattr(result, 'failures'):
                    test_failed = any(test is self
                                      for test, _ in (result.failures
                                                      + result.errors))
                    status = 'FAILED' if test_failed else 'PASSED'
                elif hasattr(outcome, 'errors'):
                    # Python 3.11+: inspect ``errors`` list of (ctx, exc_info).
                    errors = getattr(outcome, 'errors', [])
                    test_failed = any(exc_info is not None
                                      for _, exc_info in errors)
                    status = 'FAILED' if test_failed else 'PASSED'
                # else: running under pytest or other runner — cannot detect
            print(f'[TEST {status}] {self.id()}')
