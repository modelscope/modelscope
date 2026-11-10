# Copyright (c) Alibaba, Inc. and its affiliates.
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from modelscope_hub.config import HubConfig
from tests.hub.test_hub_retry import _response

from modelscope.hub.api import HubApi
from modelscope.hub.callback import ProgressCallback
from modelscope.hub.file_download import model_file_download
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.utils.constant import ModelFile

download_model_file_name = 'test.bin'
download_model_file_name2 = 'test2.bin'


class _RepoTransport:
    """Read-only wire fixtures shared by revision and metadata contract tests."""

    endpoint = 'https://modelscope.test'
    repo_id = 'test/revision-model'

    def __init__(self):
        self.requests = []
        self.files = {'master': {'README.md': b'fixture readme\n'}}
        self.model = {
            'id': 123,
            'name': 'revision-model',
            'owner': 'test',
            'description': 'fixture model',
            'private': False,
            'downloads': 12,
            'model_category': 'aigc',
            'versions': [{
                'version_id': 7
            }]
        }
        self.dataset = {
            'id': 456,
            'name': 'revision-model',
            'owner': 'test',
            'description': 'fixture dataset',
            'private': False,
            'dataset_category': 'science',
            'primary_subject_area': 'biology'
        }

    def send(self, request, **kwargs):
        self.requests.append(request)
        parsed = urlparse(request.url)
        if parsed.hostname != 'modelscope.test' or request.method not in (
                'GET', 'HEAD'):
            raise AssertionError(
                f'Unexpected request: {request.method} {request.url}')
        path = parsed.path
        query = parse_qs(parsed.query)
        if path == '/openapi/v1/models/' + self.repo_id:
            data = self.model
        elif path == '/openapi/v1/datasets/' + self.repo_id:
            data = self.dataset
        elif path == '/api/v1/models/' + self.repo_id + '/revisions':
            tags = []
            for index, revision in enumerate(self.files):
                if revision != 'master':
                    tags.append({'Revision': revision, 'CreatedAt': index + 2})
            data = {
                'RevisionMap': {
                    'Branches': [{
                        'Revision': 'master',
                        'CreatedAt': 1
                    }],
                    'Tags': tags
                }
            }
        elif path == '/api/v1/models/' + self.repo_id + '/repo/files':
            files = self.files[query.get('Revision', ['master'])[0]]
            data = {
                'Files': [{
                    'Path': name,
                    'Size': len(content),
                    'Type': 'blob',
                    'Sha256': hashlib.sha256(content).hexdigest()
                } for name, content in files.items()]
            }
        elif path == '/api/v1/models/' + self.repo_id + '/repo':
            files = self.files[query.get('Revision', ['master'])[0]]
            content = files[query['FilePath'][0]]
            size = len(content)
            requested_range = request.headers.get('Range')
            status = 200
            if requested_range:
                start, end = requested_range.removeprefix('bytes=').split('-')
                start, end = int(start), int(end) if end else size - 1
                content = content[start:end + 1]
                status = 206
            response = _response(request, status, content)
            if requested_range:
                response.headers[
                    'Content-Range'] = f'bytes {start}-{end}/{size}'
            response.headers['Accept-Ranges'] = 'bytes'
            return response
        else:
            raise AssertionError(f'Unexpected URL: {request.url}')
        return _response(
            request, body={
                'Code': 200,
                'Success': True,
                'Data': data
            })


class HubRevisionTest(unittest.TestCase):
    """Exercise real shim, compat and downloads without remote git writes."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.transport = _RepoTransport()
        config = HubConfig(
            endpoint=self.transport.endpoint,
            token='',
            config_dir=Path(self.temp.name) / 'home',
            cache_dir=Path(self.temp.name) / 'hub')
        for patcher in [
                patch('modelscope_hub.config._default_config', config),
                patch(
                    'requests.sessions.Session.send',
                    side_effect=self.transport.send)
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.api = HubApi(endpoint=self.transport.endpoint, token='')
        self.model_id = self.transport.repo_id
        self.revision = 'v0.1_test_revision'
        self.revision2 = 'v0.2_test_revision'

    def prepare_repo_data(self):
        self.transport.files['master'][
            download_model_file_name] = b'first revision\n'
        self.transport.files[self.revision] = dict(
            self.transport.files['master'])

    def _snapshot(self, revision=None, **kwargs):
        return Path(
            snapshot_download(
                self.model_id,
                revision,
                cache_dir=self.temp.name,
                endpoint=self.transport.endpoint,
                max_workers=1,
                **kwargs))

    def test_no_tag(self):
        snapshot = self._snapshot()
        self.assertEqual((snapshot / ModelFile.README).read_bytes(),
                         b'fixture readme\n')
        self.assertEqual(self._snapshot('master'), snapshot)
        path = model_file_download(
            self.model_id,
            ModelFile.README,
            cache_dir=self.temp.name,
            endpoint=self.transport.endpoint)
        self.assertEqual(Path(path).read_bytes(), b'fixture readme\n')

    def test_with_only_one_tag(self):
        self.prepare_repo_data()
        detail = self.api.get_valid_revision_detail(
            self.model_id, release_timestamp=None)
        self.assertEqual(detail['Revision'], 'master')
        snapshot = self._snapshot()
        self.assertEqual((snapshot / download_model_file_name).read_bytes(),
                         b'first revision\n')
        self.assertEqual(
            self._snapshot(self.revision),
            self._snapshot(self.revision, local_files_only=True))

    def add_new_file_and_tag(self):
        self.transport.files['master'][
            download_model_file_name2] = b'second revision\n'
        self.transport.files[self.revision2] = dict(
            self.transport.files['master'])

    def test_snapshot_download_different_revision(self):
        self.prepare_repo_data()
        first = self._snapshot(self.revision)
        self.add_new_file_and_tag()
        second = self._snapshot(self.revision2)
        self.assertNotEqual(first, second)
        self.assertEqual((first / download_model_file_name).read_bytes(),
                         b'first revision\n')
        self.assertFalse((first / download_model_file_name2).exists())
        self.assertEqual((second / download_model_file_name2).read_bytes(),
                         b'second revision\n')
        requests_before = len(self.transport.requests)
        self.assertEqual(
            self._snapshot(self.revision, local_files_only=True), first)
        self.assertEqual(len(self.transport.requests), requests_before)

    def test_file_revision_lookup_uses_explicit_endpoint_and_token(self):
        config = HubConfig(
            endpoint='https://default.test',
            token='default-fixture',
            config_dir=Path(self.temp.name) / 'other-home',
            cache_dir=Path(self.temp.name) / 'other-hub')
        with patch('modelscope_hub.config._default_config', config):
            path = model_file_download(
                self.model_id,
                ModelFile.README,
                cache_dir=self.temp.name,
                endpoint=self.transport.endpoint,
                token='explicit-fixture')
        self.assertEqual(Path(path).read_bytes(), b'fixture readme\n')
        self.assertTrue(self.transport.requests)
        self.assertTrue(
            all(
                urlparse(request.url).hostname == 'modelscope.test'
                for request in self.transport.requests))
        revision_requests = [
            request for request in self.transport.requests
            if urlparse(request.url).path.endswith('/revisions')
        ]
        self.assertTrue(revision_requests)
        authorization_headers = [
            request.headers.get('Authorization')
            for request in revision_requests
        ]
        self.assertEqual(authorization_headers,
                         ['Bearer explicit-fixture'] * len(revision_requests))
        self.assertEqual(config.token, 'default-fixture')

    def test_snapshot_progress_callbacks_are_executed(self):
        events = []

        class Capture(ProgressCallback):

            def __init__(self, filename, file_size):
                super().__init__(filename, file_size)
                events.append(('start', file_size))

            def update(self, size):
                events.append(('update', size))

            def end(self):
                events.append(('end', self.filename))

        snapshot = self._snapshot(progress_callbacks=[Capture])
        content = b'fixture readme\n'
        self.assertEqual((snapshot / ModelFile.README).read_bytes(), content)
        self.assertEqual(events[0], ('start', len(content)))
        self.assertEqual(
            sum(value for kind, value in events if kind == 'update'),
            len(content))
        self.assertEqual(events[-1], ('end', ModelFile.README))

    def test_sdk_cli_download_uses_hub_engine(self):
        from modelscope.cli.cli import run_cmd

        output = Path(self.temp.name) / 'cli-download'
        arguments = [
            'modelscope', 'download', self.model_id, '--local-dir',
            str(output), '--max-workers', '1'
        ]
        with patch('sys.argv',
                   arguments), self.assertRaises(SystemExit) as caught:
            run_cmd()
        self.assertEqual(caught.exception.code, 0)
        self.assertEqual((output / ModelFile.README).read_bytes(),
                         b'fixture readme\n')

    # Removed: test_file_download_different_revision
    # Reason: Flaky due to server-side git sync delay (tag created but files
    # not immediately accessible, causing 404 error code=10990101007).
    # This tests server timing behavior, not SDK logic.


if __name__ == '__main__':
    unittest.main()
