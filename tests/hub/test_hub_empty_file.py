# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path
import shutil
import tempfile
import unittest

from modelscope import snapshot_download


class HubEmptyFile(unittest.TestCase):

    def setUp(self):
        temporary_dir = tempfile.mkdtemp()
        self.work_dir = temporary_dir

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_download_empty_file(self):
        model_dir = snapshot_download('AI-ModelScope/GroundingDINO', cache_dir=self.work_dir)
        self.assertTrue(model_dir is not None)
        self.assertTrue(os.path.exists(os.path.join(model_dir, '1.txt')))
        self.assertTrue(os.path.exists(os.path.join(model_dir, 'configuration.json')))
        self.assertTrue(os.path.exists(os.path.join(model_dir, 'init.py')))
        self.assertTrue(os.path.exists(os.path.join(model_dir, 'README.md')))


if __name__ == '__main__':
    unittest.main()
