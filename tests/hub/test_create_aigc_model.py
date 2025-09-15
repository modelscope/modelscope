# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import tempfile
import unittest
import uuid

from requests.exceptions import HTTPError

from modelscope import HubApi
from modelscope.hub.utils.aigc import AigcModel
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import (TEST_ACCESS_TOKEN1, TEST_MODEL_ORG,
                                         delete_credential, test_level)

logger = get_logger()


class TestCreateAigcModel(unittest.TestCase):

    def setUp(self):
        self.api = HubApi()
        self.repo_id: str = f'{TEST_MODEL_ORG}/test_create_aigc_model_{uuid.uuid4().hex[-6:]}'

        # Create a dummy file for AIGC model test
        self.tmp_file = tempfile.NamedTemporaryFile(
            suffix='.safetensors', delete=False)
        self.tmp_file.write(b'This is a dummy weights file for testing.')
        self.tmp_file.close()
        self.tmp_file_path = self.tmp_file.name

    def tearDown(self):
        # Login before cleaning up, ensuring token is valid for deletion.
        try:
            self.api.login(TEST_ACCESS_TOKEN1)
            self.api.delete_model(model_id=self.repo_id)
        except HTTPError:
            pass  # It's ok if the repo doesn't exist (e.g., creation failed)
        os.remove(self.tmp_file_path)
        delete_credential()

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_create_aigc_model_expects_sha256_error(self):
        """Test creating an AIGC model repository.

        This test is expected to fail with a 'sha256 not exits' error from the server.
        This is the correct behavior when the server does not know the file yet.
        This test verifies that the SDK is correctly forming and sending the request.
        """
        logger.info(f'TEST: Attempting to create AIGC repo {self.repo_id} ...')

        # Login just before making the authenticated call.
        self.api.login(TEST_ACCESS_TOKEN1)

        # 1. Create AigcModel instance from a local file
        aigc_model = AigcModel(
            model_path=self.tmp_file_path,
            aigc_type='Checkpoint',
            base_model_type='SD_XL',
        )

        # 2. Attempt to create the model repository.
        # We expect an HTTPError because the server API requires the file's sha256
        # to be known before creating the repo.
        with self.assertRaises(HTTPError) as cm:
            self.api.create_model(
                model_id=self.repo_id,
                aigc_model=aigc_model,
            )

        # Check if the error message is the one we expect.
        # The actual error might be 'namespace is not valid' if run outside CI
        # or 'sha256 not exits' if namespace is valid. Both are acceptable failures
        # proving the SDK sent the request correctly.
        error_str = str(cm.exception)
        is_expected_error = 'sha256 not exits' in error_str or 'namespace' in error_str and 'is not valid' in error_str
        self.assertTrue(is_expected_error,
                        f'Unexpected error message: {error_str}')
        logger.info(f'TEST: Received expected error: {error_str}')
