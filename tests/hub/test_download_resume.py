import os
import tempfile
import unittest
from unittest.mock import patch

import requests

from modelscope.hub import ProgressCallback
from modelscope.hub.file_download import (download_part_with_retry,
                                          http_get_model_file)


class RecordingProgressCallback(ProgressCallback):
    instances = []

    def __init__(self, filename: str, file_size: int, resume_size: int = 0):
        super().__init__(filename, file_size)
        self.resume_size = resume_size
        self.updates = []
        self.ended = False
        self.__class__.instances.append(self)

    def update(self, size: int):
        self.updates.append(size)

    def end(self):
        self.ended = True


class FakeResponse:

    def __init__(self, chunks, error=None):
        self._chunks = chunks
        self._error = error

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=1):
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error


class DownloadResumeTest(unittest.TestCase):

    def setUp(self):
        RecordingProgressCallback.instances = []

    @patch('modelscope.hub.file_download.Retry.sleep', return_value=None)
    @patch('modelscope.hub.file_download.requests.get')
    def test_resume_initializes_progress_without_recounting(self, get_mock,
                                                            _sleep_mock):
        get_mock.return_value = FakeResponse([b'de'])

        with tempfile.TemporaryDirectory() as local_dir:
            file_path = os.path.join(local_dir, 'test.bin')
            with open(file_path, 'wb') as handle:
                handle.write(b'abc')

            http_get_model_file(
                url='http://test',
                local_dir=local_dir,
                file_name='test.bin',
                file_size=5,
                headers={},
                cookies=None,
                disable_tqdm=True,
                progress_callbacks=[RecordingProgressCallback])

            callback = RecordingProgressCallback.instances[-1]
            self.assertEqual(callback.resume_size, 3)
            self.assertEqual(callback.updates, [2])
            self.assertTrue(callback.ended)
            with open(file_path, 'rb') as handle:
                self.assertEqual(handle.read(), b'abcde')

    @patch('modelscope.hub.file_download.Retry.sleep', return_value=None)
    @patch('modelscope.hub.file_download.requests.get')
    def test_retry_keeps_progress_bound_to_new_bytes_only(self, get_mock,
                                                          _sleep_mock):
        get_mock.side_effect = [
            FakeResponse(
                [b'a'],
                error=requests.exceptions.ConnectionError('connection reset')),
            FakeResponse([b'b']),
        ]

        with tempfile.TemporaryDirectory() as local_dir:
            file_path = os.path.join(local_dir, 'test.bin')

            http_get_model_file(
                url='http://test',
                local_dir=local_dir,
                file_name='test.bin',
                file_size=2,
                headers={},
                cookies=None,
                disable_tqdm=True,
                progress_callbacks=[RecordingProgressCallback])

            callback = RecordingProgressCallback.instances[-1]
            self.assertEqual(callback.resume_size, 0)
            self.assertEqual(callback.updates, [1, 1])
            self.assertTrue(callback.ended)
            with open(file_path, 'rb') as handle:
                self.assertEqual(handle.read(), b'ab')

    @patch('modelscope.hub.file_download.Retry.sleep', return_value=None)
    @patch('modelscope.hub.file_download.requests.get')
    def test_parallel_part_resume_skips_existing_bytes(self, get_mock,
                                                       _sleep_mock):
        get_mock.return_value = FakeResponse([b'de'])

        with tempfile.TemporaryDirectory() as local_dir:
            model_file_path = os.path.join(local_dir, 'test.bin')
            part_file_path = model_file_path + '_0_4'
            with open(part_file_path, 'wb') as handle:
                handle.write(b'abc')

            callback = RecordingProgressCallback('test.bin', 5)
            download_part_with_retry((model_file_path, [callback], 0, 4,
                                      'http://test', 'test.bin', None, {}))

            self.assertEqual(callback.updates, [2])
            with open(part_file_path, 'rb') as handle:
                self.assertEqual(handle.read(), b'abcde')


if __name__ == '__main__':
    unittest.main()
