# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest
from http.client import HTTPMessage, HTTPResponse
from io import StringIO
from unittest.mock import Mock, patch

import requests
from urllib3.exceptions import MaxRetryError

from modelscope.hub.api import HubApi
from modelscope.hub.file_download import http_get_file


class HubOperationTest(unittest.TestCase):

    def setUp(self):
        self.api = HubApi()
        self.model_id = 'damo/ofa_text-to-image-synthesis_coco_large_en'

    @patch('urllib3.connectionpool.HTTPConnectionPool._get_conn')
    def test_retry_exception(self, getconn_mock):
        getconn_mock.return_value.getresponse.side_effect = [
            Mock(status=500, msg=HTTPMessage()),
            Mock(status=502, msg=HTTPMessage()),
            Mock(status=500, msg=HTTPMessage()),
        ]
        with self.assertRaises(requests.exceptions.RetryError):
            self.api.get_model_files(
                model_id=self.model_id,
                recursive=True,
            )

    @patch('urllib3.connectionpool.HTTPConnectionPool._get_conn')
    def test_retry_and_success(self, getconn_mock):
        response_body = '{"Code": 200, "Data": { "Files": [ {"CommitMessage": \
            "update","CommittedDate": 1667548386,"CommitterName": "行嗔","InCheck": false, \
            "IsLFS": false, "Mode": "33188", "Name": "README.md", "Path": "README.md", \
            "Revision": "e45fcc158894f18a7a8cfa3caf8b3dd1a2b26dc9",\
            "Sha256": "8bf99f410ae0a572e5a4a85a3949ad268d49023e5c6ef200c9bd4307f9ed0660", \
            "Size": 6399,  "Type": "blob" } ] }, "Message": "success",\
            "RequestId": "8c2a8249-ce50-49f4-85ea-36debf918714","Success": true}'

        first = 0

        def get_content(p):
            nonlocal first
            if first > 0:
                return None
            else:
                first += 1
            return response_body.encode('utf-8')

        rsp = HTTPResponse(getconn_mock)
        rsp.status = 200
        rsp.msg = HTTPMessage()
        rsp.read = get_content
        rsp.chunked = False
        # retry 2 times and success.
        getconn_mock.return_value.getresponse.side_effect = [
            Mock(status=500, msg=HTTPMessage()),
            Mock(
                status=502,
                msg=HTTPMessage(),
                body=response_body,
                read=StringIO(response_body)),
            rsp,
        ]
        model_files = self.api.get_model_files(
            model_id=self.model_id,
            recursive=True,
        )
        assert len(model_files) > 0

    @patch('urllib3.connectionpool.HTTPConnectionPool._get_conn')
    def test_retry_broken_continue(self, getconn_mock):
        test_file_name = 'video_inpainting_test.mp4'
        fp = 0

        def get_content(content_length):
            nonlocal fp
            with open('data/test/videos/%s' % test_file_name, 'rb') as f:
                f.seek(fp)
                content = f.read(content_length)
                fp += len(content)
                return content

        success_rsp = HTTPResponse(getconn_mock)
        success_rsp.status = 200
        success_rsp.msg = HTTPMessage()
        success_rsp.msg.add_header('Content-Length', '2957783')
        success_rsp.read = get_content
        success_rsp.chunked = True

        failed_rsp = HTTPResponse(getconn_mock)
        failed_rsp.status = 502
        failed_rsp.msg = HTTPMessage()
        failed_rsp.msg.add_header('Content-Length', '2957783')
        failed_rsp.read = get_content
        failed_rsp.chunked = True

        # retry 5 times and success.
        getconn_mock.return_value.getresponse.side_effect = [
            failed_rsp,
            failed_rsp,
            failed_rsp,
            failed_rsp,
            failed_rsp,
            success_rsp,
        ]
        url = 'http://www.modelscope.cn/api/v1/models/%s' % test_file_name
        http_get_file(
            url=url,
            local_dir='./',
            file_name=test_file_name,
            headers={},
            cookies=None)

        assert os.path.exists('./%s' % test_file_name)
        os.remove('./%s' % test_file_name)

    @patch('urllib3.connectionpool.HTTPConnectionPool._get_conn')
    def test_retry_broken_continue_retry_failed(self, getconn_mock):
        test_file_name = 'video_inpainting_test.mp4'
        fp = 0

        def get_content(content_length):
            nonlocal fp
            with open('data/test/videos/%s' % test_file_name, 'rb') as f:
                f.seek(fp)
                content = f.read(content_length)
                fp += len(content)
                return content

        failed_rsp = HTTPResponse(getconn_mock)
        failed_rsp.status = 502
        failed_rsp.msg = HTTPMessage()
        failed_rsp.msg.add_header('Content-Length', '2957783')
        failed_rsp.read = get_content
        failed_rsp.chunked = True

        # retry 6 times and success.
        getconn_mock.return_value.getresponse.side_effect = [
            failed_rsp,
            failed_rsp,
            failed_rsp,
            failed_rsp,
            failed_rsp,
            failed_rsp,
        ]
        url = 'http://www.modelscope.cn/api/v1/models/%s' % test_file_name
        with self.assertRaises(MaxRetryError):
            http_get_file(
                url=url,
                local_dir='./',
                file_name=test_file_name,
                headers={},
                cookies=None)

        assert not os.path.exists('./%s' % test_file_name)


if __name__ == '__main__':
    unittest.main()
