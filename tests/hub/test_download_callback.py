import tempfile
import unittest
from unittest import mock

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


class SnapshotDownloadForwardTest(unittest.TestCase):
    """Network-free tests: the shim forwards progress_callbacks to compat."""

    # ``modelscope.hub.snapshot_download`` the attribute is shadowed by the
    # re-exported function, so patch via the fully qualified module string.
    _COMPAT_TARGET = \
        'modelscope.hub.snapshot_download._compat_snapshot_download'

    def test_progress_callbacks_forwarded_to_compat(self):
        from modelscope.hub.snapshot_download import snapshot_download

        with mock.patch(
                self._COMPAT_TARGET, return_value='/tmp/snapshot') as m:
            result = snapshot_download(
                'owner/repo', progress_callbacks=[NewProgressCallback])

        self.assertEqual(result, '/tmp/snapshot')
        _, kwargs = m.call_args
        self.assertEqual(kwargs['progress_callbacks'], [NewProgressCallback])

    def test_progress_callbacks_default_none(self):
        from modelscope.hub.snapshot_download import snapshot_download

        with mock.patch(
                self._COMPAT_TARGET, return_value='/tmp/snapshot') as m:
            snapshot_download('owner/repo')

        _, kwargs = m.call_args
        self.assertIsNone(kwargs['progress_callbacks'])


if __name__ == '__main__':
    unittest.main()
