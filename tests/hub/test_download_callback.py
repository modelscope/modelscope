import tempfile
import unittest

from tqdm import tqdm

from modelscope import snapshot_download
from modelscope.hub import ProgressCallback


class NewProgressCallback(ProgressCallback):
    all_files = set()  # just for test

    def __init__(self, filename: str, file_size: int):
        super().__init__(filename, file_size)
        self.progress = tqdm(total=file_size)
        self.all_files.add(filename)

    def update(self, size: int):
        self.progress.update(size)

    def end(self):
        self.all_files.remove(self.filename)
        assert self.progress.n == self.progress.total == self.file_size
        self.progress.close()


class ProgressCallbackTest(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_progress_callback(self):
        model_dir = snapshot_download(
            'swift/test_lora',
            progress_callbacks=[NewProgressCallback],
            cache_dir=self.temp_dir.name)
        print(f'model_dir: {model_dir}')
        self.assertTrue(len(NewProgressCallback.all_files) == 0)

    def test_empty_progress_callback(self):
        model_dir = snapshot_download(
            'swift/test_lora',
            progress_callbacks=[],
            cache_dir=self.temp_dir.name)
        print(f'model_dir: {model_dir}')


if __name__ == '__main__':
    unittest.main()
