# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import tempfile
import unittest
import uuid
from unittest import mock

import json

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
        except Exception as e:
            logger.warning(f'Error deleting model {self.repo_id}: {e}')
        os.remove(self.tmp_file_path)
        delete_credential()

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_create_aigc_model(self):
        """Test creating and uploading an AIGC model repository."""
        logger.info(f'TEST: Attempting to create AIGC repo {self.repo_id} ...')

        # Login just before making the authenticated call.
        self.api.login(TEST_ACCESS_TOKEN1)

        # 1. Create AigcModel instance from a local file
        aigc_model = AigcModel(
            model_path=self.tmp_file_path,
            aigc_type='Checkpoint',
            base_model_type='SD_XL',
            readme_content='# AIGC integration test\n',
        )

        # 2. Create the repository through the AIGC-specific compatibility path.
        model_url = self.api.create_model(
            model_id=self.repo_id,
            visibility=1,
            aigc_model=aigc_model,
        )

        self.assertEqual(model_url,
                         f'{self.api.endpoint}/models/{self.repo_id}')
        files = self.api.get_model_files(self.repo_id, revision='master')
        paths = {item['Path'] for item in files}
        self.assertIn(os.path.basename(self.tmp_file_path), paths)
        readme_path = self.api.download_file(
            self.repo_id,
            repo_type='model',
            file_path='README.md',
            force=True,
        )
        self.assertEqual(
            readme_path.read_text(encoding='utf-8'),
            '# AIGC integration test\n')


class TestAigcModelReadmeContent(unittest.TestCase):

    def setUp(self):
        self.tmp_file = tempfile.NamedTemporaryFile(
            suffix='.safetensors', delete=False)
        self.tmp_file.write(b'dummy weights')
        self.tmp_file.close()

    def tearDown(self):
        os.remove(self.tmp_file.name)

    def test_optional_readme_content(self):
        readme_content = '# AIGC v1.3\n\nCustom model card.\n'
        aigc_model = AigcModel(
            model_path=self.tmp_file.name,
            aigc_type='LoRA',
            base_model_type='SD_XL',
            readme_content=readme_content,
        )

        self.assertEqual(aigc_model.readme_content, readme_content)
        self.assertEqual(aigc_model.to_dict()['readme_content'],
                         readme_content)

    def test_readme_content_defaults_to_none(self):
        aigc_model = AigcModel(
            model_path=self.tmp_file.name,
            aigc_type='LoRA',
            base_model_type='SD_XL',
        )

        self.assertIsNone(aigc_model.readme_content)

    def test_from_json_file_accepts_readme_content(self):
        config_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False, encoding='utf-8')
        self.addCleanup(os.remove, config_file.name)
        readme_content = '# Loaded from JSON\n'
        json.dump(
            {
                'model_path': self.tmp_file.name,
                'aigc_type': 'LoRA',
                'base_model_type': 'SD_XL',
                'base_model_id': 'owner/base-model',
                'readme_content': readme_content,
            }, config_file)
        config_file.close()

        aigc_model = AigcModel.from_json_file(config_file.name)

        self.assertEqual(aigc_model.readme_content, readme_content)

    def test_readme_content_must_be_string(self):
        with self.assertRaisesRegex(TypeError,
                                    'readme_content must be a string or None'):
            AigcModel(
                model_path=self.tmp_file.name,
                aigc_type='LoRA',
                base_model_type='SD_XL',
                readme_content=123,
            )

    @mock.patch('modelscope.hub.utils.aigc.requests.put')
    def test_preupload_uses_shared_blob_timeout_by_default(self, put):
        put.return_value.json.return_value = {}
        aigc_model = AigcModel(
            model_path=self.tmp_file.name,
            aigc_type='LoRA',
            base_model_type='SD_XL',
        )

        aigc_model.preupload_weights(
            cookies={'m_session_id': 'ms-test'}, headers={})

        self.assertEqual(put.call_args.kwargs['timeout'], (30, 3600))

    @mock.patch('modelscope.hub.utils.aigc.requests.put')
    def test_preupload_preserves_explicit_timeout(self, put):
        put.return_value.json.return_value = {}
        aigc_model = AigcModel(
            model_path=self.tmp_file.name,
            aigc_type='LoRA',
            base_model_type='SD_XL',
        )

        aigc_model.preupload_weights(
            cookies={'m_session_id': 'ms-test'}, headers={}, timeout=17)

        self.assertEqual(put.call_args.kwargs['timeout'], 17)

    def test_legacy_upload_constants_match_modelscope_hub(self):
        from modelscope.hub import constants as legacy
        from modelscope_hub import constants as canonical

        self.assertEqual(legacy.UPLOAD_BLOB_TIMEOUT,
                         (canonical.UPLOAD_BLOB_CONNECT_TIMEOUT_SECONDS,
                          canonical.UPLOAD_BLOB_READ_TIMEOUT_SECONDS))
        self.assertEqual(legacy.UPLOAD_BLOB_MAX_RETRIES,
                         canonical.UPLOAD_BLOB_MAX_ATTEMPTS)
        self.assertEqual(legacy.UPLOAD_SIZE_THRESHOLD_TO_ENFORCE_LFS,
                         canonical.UPLOAD_LFS_FORCE_THRESHOLD_BYTES)
        self.assertEqual(legacy.UPLOAD_REACT_ROUND3_FILE_DELAY,
                         canonical.UPLOAD_RECOVERY_SINGLE_FILE_DELAY_SECONDS)
