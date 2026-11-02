# Copyright (c) Alibaba, Inc. and its affiliates.
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import json
import requests
from modelscope_hub.constants import REPO_TREE_PAGE_MAX_ATTEMPTS
from urllib3.exceptions import MaxRetryError

from modelscope.hub.api import HubApi
from modelscope.hub.constants import API_FILE_DOWNLOAD_RETRY_TIMES
from modelscope.hub.errors import ServerError
from modelscope.hub.file_download import http_get_model_file


def _response(request, status=200, body=b''):
    """Build a real response with byte-stream semantics for transport fixtures."""
    if isinstance(body, bytes):
        content = body
    else:
        content = json.dumps(body).encode('utf-8')
    response = requests.Response()
    response.status_code = status
    response.url = request.url
    response.request = request
    response.headers['Content-Length'] = str(len(content))
    response._content = content
    response._content_consumed = True
    response.raw = io.BytesIO(content)
    return response


class HubRetryTest(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.api = HubApi(endpoint='https://modelscope.test', token='')
        self.model_id = 'test/retry-model'

    def test_retry_exception(self):

        def send(request, **kwargs):
            return _response(request, 502, {
                'Code': 502,
                'Message': 'fixture failure'
            })

        with patch('requests.sessions.Session.send', side_effect=send) as transport, \
                patch('modelscope_hub._legacy_api.time.sleep'), \
                self.assertRaises(ServerError) as caught:
            self.api.get_model_files(model_id=self.model_id, recursive=True)
        self.assertEqual(transport.call_count, REPO_TREE_PAGE_MAX_ATTEMPTS)
        self.assertEqual(caught.exception.response.status_code, 502)

    def test_retry_and_success(self):
        statuses = iter([500, 502, 200])
        body = {
            'Code': 200,
            'Success': True,
            'Data': {
                'Files': [{
                    'Path': 'README.md',
                    'Size': 16,
                    'Type': 'blob'
                }]
            }
        }

        def send(request, **kwargs):
            status = next(statuses)
            payload = body if status == 200 else {
                'Code': status,
                'Message': 'fixture failure'
            }
            return _response(request, status, payload)

        with patch('requests.sessions.Session.send', side_effect=send) as transport, \
                patch('modelscope_hub._legacy_api.time.sleep'):
            files = self.api.get_model_files(
                model_id=self.model_id, recursive=True)
        self.assertEqual(files, [{'Path': 'README.md', 'Size': 16}])
        self.assertEqual(transport.call_count, 3)
        self.assertTrue(
            all('/repo/files?' in call.args[0].url
                for call in transport.call_args_list))

    def test_retry_broken_continue(self):
        content = b'0123456789abcdef'
        ranges = []

        def send(request, **kwargs):
            ranges.append(request.headers['Range'])
            if len(ranges) == 1:
                response = _response(request, body=content)

                def chunks(chunk_size=1, **kwargs):
                    yield content[:4]
                    raise requests.ConnectionError(
                        'fixture interrupted stream')

                response.iter_content = chunks
                return response
            return _response(request, 206, content[4:])

        with patch('requests.sessions.Session.send', side_effect=send), \
                patch('urllib3.util.retry.Retry.sleep'):
            checksum = http_get_model_file(
                url='https://modelscope.test/download',
                local_dir=self.temp.name,
                file_name='payload.bin',
                file_size=len(content),
                cookies=None,
                disable_tqdm=True)
        self.assertEqual((Path(self.temp.name) / 'payload.bin').read_bytes(),
                         content)
        self.assertEqual(ranges, ['bytes=0-15', 'bytes=4-15'])
        self.assertIsNone(checksum)

    def test_retry_broken_continue_retry_failed(self):

        def send(request, **kwargs):
            return _response(request, 502, b'fixture failure')

        with patch('requests.sessions.Session.send', side_effect=send) as transport, \
                patch('urllib3.util.retry.Retry.sleep'), self.assertRaises(MaxRetryError):
            http_get_model_file(
                url='https://modelscope.test/download',
                local_dir=self.temp.name,
                file_name='payload.bin',
                file_size=16,
                cookies=None,
                disable_tqdm=True)
        self.assertEqual(transport.call_count,
                         API_FILE_DOWNLOAD_RETRY_TIMES + 1)
        self.assertEqual((Path(self.temp.name) / 'payload.bin').stat().st_size,
                         0)


if __name__ == '__main__':
    unittest.main()
