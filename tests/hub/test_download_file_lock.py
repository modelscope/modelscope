import hashlib
import multiprocessing
import os
import tempfile
import unittest

from modelscope import snapshot_download


def download_model(model_name, cache_dir, enable_lock):
    if not enable_lock:
        os.environ['MODELSCOPE_HUB_FILE_LOCK'] = 'false'
    snapshot_download(model_name, cache_dir=cache_dir)


class FileLockDownloadingTest(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_multi_processing_file_lock(self):

        models = [
            'iic/nlp_bert_relation-extraction_chinese-base',
            'iic/nlp_bert_relation-extraction_chinese-base',
            'iic/nlp_bert_relation-extraction_chinese-base',
        ]
        args_list = [(model, self.temp_dir.name, True) for model in models]

        with multiprocessing.Pool(processes=3) as pool:
            pool.starmap(download_model, args_list)

        def get_file_sha256(file_path):
            sha256_hash = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()

        tensor_file = os.path.join(
            self.temp_dir.name, 'iic',
            'nlp_bert_relation-extraction_chinese-base', 'pytorch_model.bin')
        sha256 = '2b623d2c06c8101c1283657d35bc22d69bcc10f62ded0ba6d0606e4130f9c8af'
        self.assertTrue(get_file_sha256(tensor_file) == sha256)

    def test_multi_processing_disabled(self):
        try:
            models = [
                'iic/nlp_bert_backbone_base_std',
                'iic/nlp_bert_backbone_base_std',
                'iic/nlp_bert_backbone_base_std',
            ]
            args_list = [(model, self.temp_dir.name, False)
                         for model in models]

            with multiprocessing.Pool(processes=3) as pool:
                pool.starmap(download_model, args_list)

            def get_file_sha256(file_path):
                sha256_hash = hashlib.sha256()
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b''):
                        sha256_hash.update(chunk)
                return sha256_hash.hexdigest()

            tensor_file = os.path.join(self.temp_dir.name, 'iic',
                                       'nlp_bert_backbone_base_std',
                                       'pytorch_model.bin')
            sha256 = 'c6a293a8091f7eaa1ac7ecf88fd6f4cc00f6957188b2730d34faa787f15d3caa'
            self.assertTrue(get_file_sha256(tensor_file) != sha256)
        except Exception:  # noqa
            pass


if __name__ == '__main__':
    unittest.main()
