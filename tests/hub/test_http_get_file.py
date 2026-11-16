# Copyright (c) Alibaba, Inc. and its affiliates.
import gzip
import http.server
import os
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch

from urllib3.exceptions import MaxRetryError

from modelscope.hub.errors import FileDownloadError
from modelscope.hub.file_download import http_get_file

PAYLOAD = (b'modelscope-http-get-file-' * 2000)[:48000]
# Large enough to span several read chunks, so an interrupted download leaves
# bytes behind and the retry is sent as a range request.
BIG_PAYLOAD = bytes(range(256)) * 12000


class _Handler(http.server.BaseHTTPRequestHandler):

    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def do_GET(self):
        value = self.headers.get('Range', '')
        self.range_start = int(value[6:].split('-')[0]) if value else 0
        self.close_connection = True
        try:
            self.server.respond(self)
        except ConnectionError:
            # Invalid ranges can be rejected before the response body is read.
            pass


class HttpGetFileTest(unittest.TestCase):
    """``http_get_file`` must not reject correctly downloaded files.

    Network-free: every case talks to a local ``http.server``.
    """

    def _download(self, respond, headers=None):
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        sock.close()
        httpd = http.server.ThreadingHTTPServer(('127.0.0.1', port), _Handler)
        httpd.respond = respond
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.local_dir = tempfile.mkdtemp()
        try:
            http_get_file(
                url='http://127.0.0.1:%d/file.bin' % port,
                local_dir=self.local_dir,
                file_name='file.bin',
                cookies=None,
                headers={} if headers is None else headers)
            with open(os.path.join(self.local_dir, 'file.bin'), 'rb') as f:
                return f.read()
        finally:
            httpd.shutdown()
            httpd.server_close()

    def _assert_failed(self, respond):
        with patch(
                'modelscope.hub.file_download.'
                'API_FILE_DOWNLOAD_RETRY_TIMES', 1):
            with self.assertRaises((FileDownloadError, MaxRetryError)):
                self._download(respond)
        self.assertFalse(
            os.path.exists(os.path.join(self.local_dir, 'file.bin')))

    @staticmethod
    def _write_chunked(handler, body):
        for offset in range(0, len(body), 8000):
            part = body[offset:offset + 8000]
            handler.wfile.write(b'%x\r\n' % len(part) + part + b'\r\n')
        handler.wfile.write(b'0\r\n\r\n')

    def _resuming(self,
                  restart_from_zero=False,
                  chunked=False,
                  ignore_range=False):
        """Serve BIG_PAYLOAD, dropping the connection on the first attempt."""
        state = {'requests': 0}
        lock = threading.Lock()

        def respond(handler):
            with lock:
                state['requests'] += 1
                attempt = state['requests']
            resumed = handler.range_start > 0
            body = BIG_PAYLOAD[0 if restart_from_zero or ignore_range else
                               handler.range_start:]
            handler.send_response(206 if resumed and not ignore_range else 200)
            if chunked and resumed:
                handler.send_header('Transfer-Encoding', 'chunked')
            else:
                handler.send_header('Content-Length', str(len(body)))
            if resumed and not ignore_range:
                handler.send_header(
                    'Content-Range', 'bytes %d-%d/%d' %
                    (handler.range_start, len(BIG_PAYLOAD) - 1,
                     len(BIG_PAYLOAD)))
            handler.end_headers()
            if attempt == 1:
                handler.wfile.write(body[:len(body) // 2])
                handler.close_connection = True
            elif chunked:
                self._write_chunked(handler, body)
            else:
                handler.wfile.write(body)

        return respond, state

    def test_content_encoding_gzip(self):
        # `Content-Length` counts the compressed bytes, requests writes out
        # the transparently decoded ones.
        body = gzip.compress(PAYLOAD)

        def respond(handler):
            handler.send_response(200)
            handler.send_header('Content-Encoding', 'gzip')
            handler.send_header('Content-Length', str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)

        self.assertEqual(self._download(respond), PAYLOAD)

    def test_without_content_length(self):
        # A chunked response carries no `Content-Length` at all.
        def respond(handler):
            handler.send_response(200)
            handler.send_header('Transfer-Encoding', 'chunked')
            handler.end_headers()
            for i in range(0, len(PAYLOAD), 8000):
                part = PAYLOAD[i:i + 8000]
                handler.wfile.write(b'%x\r\n' % len(part) + part + b'\r\n')
            handler.wfile.write(b'0\r\n\r\n')

        self.assertEqual(self._download(respond), PAYLOAD)

    def test_resume_with_partial_content(self):
        # The retry is answered with `206`, whose `Content-Length` excludes
        # the bytes already on disk.
        respond, state = self._resuming()
        self.assertEqual(self._download(respond), BIG_PAYLOAD)
        self.assertGreater(state['requests'], 1)

    def test_chunked_partial_content_must_cover_complete_file(self):

        def respond(handler):
            handler.send_response(206)
            handler.send_header('Transfer-Encoding', 'chunked')
            handler.send_header('Content-Range',
                                'bytes 0-99/%d' % len(BIG_PAYLOAD))
            handler.end_headers()
            self._write_chunked(handler, BIG_PAYLOAD[:100])

        self._assert_failed(respond)

    def test_resume_with_chunked_partial_content(self):
        respond, state = self._resuming(chunked=True)
        self.assertEqual(self._download(respond), BIG_PAYLOAD)
        self.assertGreater(state['requests'], 1)

    def test_resume_ignoring_range_restarts_from_zero(self):
        for chunked in (False, True):
            with self.subTest(chunked=chunked):
                respond, state = self._resuming(
                    chunked=chunked, ignore_range=True)
                self.assertEqual(self._download(respond), BIG_PAYLOAD)
                self.assertGreater(state['requests'], 1)

    def test_partial_content_requires_verifiable_complete_suffix(self):
        for content_range in (None, 'items 0-47999/48000',
                              'bytes 1-47999/48000', 'bytes 0-47999/*',
                              'bytes 0-48000/48000'):
            with self.subTest(content_range=content_range):

                def respond(handler):
                    handler.send_response(206)
                    handler.send_header('Transfer-Encoding', 'chunked')
                    if content_range is not None:
                        handler.send_header('Content-Range', content_range)
                    handler.end_headers()
                    self._write_chunked(handler, PAYLOAD)

                self._assert_failed(respond)

    def test_complete_chunked_partial_content(self):

        def respond(handler):
            handler.send_response(206)
            handler.send_header('Transfer-Encoding', 'chunked')
            handler.send_header(
                'Content-Range', 'bytes 0-%d/%d' %
                (len(PAYLOAD) - 1, len(PAYLOAD)))
            handler.end_headers()
            self._write_chunked(handler, PAYLOAD)

        self.assertEqual(self._download(respond), PAYLOAD)

    def test_encoded_partial_content_cannot_use_decoded_offsets(self):
        body = gzip.compress(PAYLOAD)

        def respond(handler):
            handler.send_response(206)
            handler.send_header('Content-Encoding', 'gzip')
            handler.send_header('Content-Length', str(len(body)))
            handler.send_header('Content-Range', 'bytes 0-%d/%d' %
                                (len(body) - 1, len(body)))
            handler.end_headers()
            handler.wfile.write(body)

        self._assert_failed(respond)

    def test_partial_content_defaults_to_identity_encoding(self):
        received_encodings = []

        def respond(handler):
            encoding = handler.headers.get('Accept-Encoding')
            received_encodings.append(encoding)
            body = PAYLOAD if encoding == 'identity' else gzip.compress(
                PAYLOAD)
            handler.send_response(206)
            if encoding != 'identity':
                handler.send_header('Content-Encoding', 'gzip')
            handler.send_header('Content-Length', str(len(body)))
            handler.send_header('Content-Range', 'bytes 0-%d/%d' %
                                (len(body) - 1, len(body)))
            handler.end_headers()
            handler.wfile.write(body)

        with patch(
                'modelscope.hub.file_download.'
                'API_FILE_DOWNLOAD_RETRY_TIMES', 1):
            self.assertEqual(self._download(respond), PAYLOAD)
        self.assertEqual(received_encodings, ['identity'])

    def test_caller_encoding_header_is_preserved_case_insensitively(self):
        for name in ('Accept-Encoding', 'accept-encoding', 'aCcEpT-EnCoDiNg'):
            with self.subTest(header_name=name):
                received_encodings = []
                headers = {name: 'gzip'}
                body = gzip.compress(PAYLOAD)

                def respond(handler):
                    received_encodings.append(
                        handler.headers.get_all('Accept-Encoding'))
                    handler.send_response(200)
                    handler.send_header('Content-Encoding', 'gzip')
                    handler.send_header('Content-Length', str(len(body)))
                    handler.end_headers()
                    handler.wfile.write(body)

                self.assertEqual(self._download(respond, headers), PAYLOAD)
                self.assertEqual(received_encodings, [['gzip']])
                self.assertEqual(headers, {name: 'gzip'})

    def test_partial_content_from_wrong_offset_fails(self):
        # A `206` that restarts from byte 0 appends a duplicate copy; the
        # complete length in `Content-Range` still catches it.
        self._assert_failed(self._resuming(restart_from_zero=True)[0])

    def test_truncated_response_still_fails(self):

        def respond(handler):
            handler.send_response(200)
            handler.send_header('Content-Length', str(len(PAYLOAD)))
            handler.end_headers()
            handler.wfile.write(PAYLOAD[:1000])
            handler.close_connection = True

        self._assert_failed(respond)


if __name__ == '__main__':
    unittest.main()
