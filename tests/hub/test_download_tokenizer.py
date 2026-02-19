# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path
import shutil
import tempfile
import unittest

from modelscope import snapshot_download


class TestDownloadTokenizer(unittest.TestCase):

    def setUp(self):
        temporary_dir = tempfile.mkdtemp()
        self.work_dir = temporary_dir

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_download_tokenizer(self):
        ignore_file_pattern = [
            r'*.bin',
            r'*.safetensors',
            r'*.pth',
            r'*.pt',
            r'*.h5',
            r'*.ckpt',
            r'*.zip',
            r'*.onnx',
            r'*.tar',
            r'*.gz',
        ]
        model_dir = snapshot_download(
            'Qwen/Qwen3-0.6B',
            cache_dir=self.work_dir,
            ignore_file_pattern=ignore_file_pattern)
        self.assertTrue(model_dir is not None)
        self.assertTrue(
            os.path.exists(os.path.join(model_dir, 'tokenizer.json')))
        self.assertFalse(
            os.path.exists(os.path.join(model_dir, 'model.safetensors')))


if __name__ == '__main__':
    unittest.main()
