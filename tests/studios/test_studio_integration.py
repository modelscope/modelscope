# Copyright (c) Alibaba, Inc. and its affiliates.
"""Integration tests that hit the real ModelScope Studio API.

These tests are guarded by ``test_level() >= 1`` and require a valid
``MODELSCOPE_API_TOKEN`` (loaded from ``tests/studios/.env`` if present).
"""
import unittest
import uuid

from tests.studios.conftest_env import get_test_config, load_test_env

from modelscope.hub.api import HubApi
from modelscope.hub.constants import Visibility
from modelscope.utils.constant import REPO_TYPE_STUDIO, StudioSDKType
from modelscope.utils.test_utils import test_level

load_test_env()


@unittest.skipUnless(test_level() >= 1,
                     'skip: requires network and valid token (TEST_LEVEL >= 1)'
                     )
class TestStudioIntegration(unittest.TestCase):
    """End-to-end tests against the live Studio OpenAPI."""

    @classmethod
    def setUpClass(cls):
        config = get_test_config()
        cls.token = config['token']
        cls.owner = config['owner']
        cls.endpoint = config['endpoint']
        cls.visibility = config['visibility']
        if not cls.token or cls.token == 'your_access_token_here':
            raise unittest.SkipTest('MODELSCOPE_API_TOKEN not set')
        cls.api = HubApi(token=cls.token, endpoint=cls.endpoint)
        cls.created_studios = []

    @classmethod
    def tearDownClass(cls):
        # Best-effort cleanup. Currently the Studio API does not expose a
        # delete endpoint; we simply log what was created.
        for sid in getattr(cls, 'created_studios', []):
            print(f'[integration] created studio left behind: {sid}')

    def setUp(self):
        # Generate a unique repo name per test; studio_id is owner + name.
        self.name = f'ut-studio-{uuid.uuid4().hex[:8]}'
        self.studio_id = f'{self.owner}/{self.name}'

    def test_create_and_check_studio(self):
        """Create a studio and verify it is visible via repo_exists."""
        # Map configured visibility string -> Visibility enum value.
        visibility_value = (
            Visibility.PRIVATE
            if self.visibility == 'private' else Visibility.PUBLIC)

        url = self.api.create_repo(
            self.studio_id,
            repo_type=REPO_TYPE_STUDIO,
            token=self.token,
            visibility=visibility_value,
            sdk_type=StudioSDKType.GRADIO,
        )
        self.assertIn(self.name, url)
        self.__class__.created_studios.append(self.studio_id)

        exists = self.api.repo_exists(
            self.studio_id, repo_type=REPO_TYPE_STUDIO, token=self.token)
        self.assertTrue(exists)

    def test_repo_exists_nonexistent(self):
        """Non-existent studios must report False, not raise."""
        bogus = f'{self.owner}/nonexistent-{uuid.uuid4().hex[:8]}'
        exists = self.api.repo_exists(
            bogus, repo_type=REPO_TYPE_STUDIO, token=self.token)
        self.assertFalse(exists)

    def test_secrets_lifecycle(self):
        """Round-trip add → list → update → delete on a real studio.

        Creates a fresh studio (using configured visibility) so the test is
        self-contained and does not rely on a pre-existing repo.
        """
        visibility_value = (
            Visibility.PRIVATE
            if self.visibility == 'private' else Visibility.PUBLIC)
        self.api.create_repo(
            self.studio_id,
            repo_type=REPO_TYPE_STUDIO,
            token=self.token,
            visibility=visibility_value,
            sdk_type=StudioSDKType.GRADIO,
        )
        self.__class__.created_studios.append(self.studio_id)

        secret_key = f'TEST_KEY_{uuid.uuid4().hex[:6]}'

        self.api.add_studio_secret(
            self.studio_id, secret_key, 'test_value', token=self.token)
        try:
            secrets = self.api.list_studio_secrets(
                self.studio_id, token=self.token)
            keys = [s.get('key') for s in secrets if isinstance(s, dict)]
            self.assertIn(secret_key, keys)

            self.api.update_studio_secret(
                self.studio_id, secret_key, 'new_value', token=self.token)
        finally:
            self.api.delete_studio_secret(
                self.studio_id, secret_key, token=self.token)

        secrets = self.api.list_studio_secrets(
            self.studio_id, token=self.token)
        keys = [s.get('key') for s in secrets if isinstance(s, dict)]
        self.assertNotIn(secret_key, keys)


if __name__ == '__main__':
    unittest.main()
