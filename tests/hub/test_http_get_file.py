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
        self.server.respond(self)


class HttpGetFileTest(unittest.TestCase):
    """``http_get_file`` must not reject correctly downloaded files.

    Network-free: every case talks to a local ``http.server``.
    """

    def _download(self, respond):
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
                headers={})
            with open(os.path.join(self.local_dir, 'file.bin'), 'rb') as f:
                return f.read()
        finally:
            httpd.shutdown()

    def _assert_failed(self, respond):
        with patch(
                'modelscope.hub.file_download.'
                'API_FILE_DOWNLOAD_RETRY_TIMES', 1):
            with self.assertRaises((FileDownloadError, MaxRetryError)):
                self._download(respond)
        self.assertFalse(
            os.path.exists(os.path.join(self.local_dir, 'file.bin')))

    def _resuming(self, restart_from_zero=False):
        """Serve BIG_PAYLOAD, dropping the connection on the first attempt."""
        state = {'requests': 0}
        lock = threading.Lock()

        def respond(handler):
            with lock:
                state['requests'] += 1
                attempt = state['requests']
            resumed = handler.range_start > 0
            body = BIG_PAYLOAD[0 if restart_from_zero else handler.
                               range_start:]
            handler.send_response(206 if resumed else 200)
            handler.send_header('Content-Length', str(len(body)))
            if resumed:
                handler.send_header(
                    'Content-Range', 'bytes %d-%d/%d' %
                    (handler.range_start, len(BIG_PAYLOAD) - 1,
                     len(BIG_PAYLOAD)))
            handler.end_headers()
            if attempt == 1:
                handler.wfile.write(body[:len(body) // 2])
                handler.close_connection = True
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
