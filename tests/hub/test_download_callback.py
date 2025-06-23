import unittest

from tqdm import tqdm

from modelscope import snapshot_download
from modelscope.hub import ProgressCallback


class NewProgressCallback(ProgressCallback):

    def __init__(self, filename: str, file_size: int):
        super().__init__(filename, file_size)
        self.progress = tqdm(total=file_size)

    def update(self, size: int):
        self.progress.update(size)

    def end(self):
        self.progress.close()


class ProgressCallbackTest(unittest.TestCase):

    def test_progress_callback(self):
        model_dir = snapshot_download(
            'Qwen/Qwen3-0.6B', progress_callbacks=[NewProgressCallback])
        print(f'model_dir: {model_dir}')

        model_dir = snapshot_download('Qwen/Qwen3-0.6B', progress_callbacks=[])
        print(f'model_dir: {model_dir}')


if __name__ == '__main__':
    unittest.main()
