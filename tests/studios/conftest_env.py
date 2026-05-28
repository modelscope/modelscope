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
    return {
        'token': os.environ.get('MODELSCOPE_API_TOKEN'),
        'owner': os.environ.get('TEST_STUDIO_OWNER', 'test_user'),
        'visibility': os.environ.get('TEST_STUDIO_VISIBILITY', 'private'),
        'endpoint': os.environ.get('MODELSCOPE_ENDPOINT',
                                   'https://modelscope.cn'),
    }
