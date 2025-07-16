import hashlib
import multiprocessing
import os
import tempfile
import unittest

from modelscope import snapshot_download


def download_model(model_name, cache_dir):
    snapshot_download(model_name, cache_dir=cache_dir)

class FileLockDownloadingTest(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_multi_processing_file_lock(self):

        models = [
            'Qwen/Qwen3-0.6B',
            'Qwen/Qwen3-0.6B',
            'Qwen/Qwen3-0.6B',
        ]
        args_list = [(model, self.temp_dir.name) for model in models]

        with multiprocessing.Pool(processes=3) as pool:
            pool.starmap(download_model, args_list)

        def get_file_sha256(file_path):
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()

        self.assertTrue(get_file_sha256(os.path.join(self.temp_dir.name, 'Qwen', 'Qwen3-0.6B', 'model.safetensors')) == 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b')


if __name__ == '__main__':
    unittest.main()
