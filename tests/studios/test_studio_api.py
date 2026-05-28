# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for HubApi Studio operations (mocked HTTP layer).

These tests do not touch the network: every test patches
``HubApi._build_bearer_headers`` and the corresponding ``self.session.<verb>``
to assert that the right URL, body and parameters are produced and that the
response is decoded correctly.
"""
import unittest
from unittest.mock import MagicMock, patch

from tests.studios.conftest_env import get_test_config

from modelscope.hub.api import HubApi
from modelscope.hub.constants import Visibility
from modelscope.hub.errors import InvalidParameter
from modelscope.utils.constant import (REPO_TYPE_STUDIO, REPO_TYPE_SUPPORT,
                                       StudioHardware, StudioSDKType,
                                       StudioStatus)


def _make_response(status_code=200, json_data=None):
    """Create a MagicMock response that satisfies ``handle_http_response``."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.reason = 'OK' if resp.ok else 'Not Found'
    resp.url = 'https://modelscope.cn/mock'
    resp.headers = {'X-Request-Id': 'req-mock'}
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    return resp


class TestStudioConstants(unittest.TestCase):
    """Studio-related constants are defined and have the expected shape."""

    def test_repo_type_support_includes_studio(self):
        self.assertIn('studio', REPO_TYPE_SUPPORT)
        self.assertEqual(REPO_TYPE_STUDIO, 'studio')

    def test_studio_sdk_types(self):
        self.assertEqual(StudioSDKType.GRADIO, 'gradio')
        self.assertEqual(StudioSDKType.STREAMLIT, 'streamlit')
        self.assertEqual(StudioSDKType.DOCKER, 'docker')
        self.assertEqual(StudioSDKType.STATIC, 'static')
        self.assertEqual(len(StudioSDKType.SUPPORTED), 4)
        self.assertIn(StudioSDKType.GRADIO, StudioSDKType.SUPPORTED)

    def test_studio_hardware(self):
        self.assertEqual(StudioHardware.DEFAULT, 'platform/2v-cpu-16g-mem')
        self.assertEqual(StudioHardware.CPU_2V_16G, 'platform/2v-cpu-16g-mem')
        self.assertEqual(len(StudioHardware.SUPPORTED), 3)
        self.assertIn(StudioHardware.DEFAULT, StudioHardware.SUPPORTED)

    def test_studio_status_values(self):
        self.assertEqual(StudioStatus.RUNNING, 'Running')
        self.assertEqual(StudioStatus.STOPPED, 'Stopped')
        self.assertEqual(StudioStatus.DEPLOYING, 'Deploying')
        self.assertEqual(StudioStatus.STOPPING, 'Stopping')


class TestParseStudioId(unittest.TestCase):
    """Validate the ``HubApi._parse_studio_id`` helper."""

    def test_valid_studio_id(self):
        owner, name = HubApi._parse_studio_id('myuser/my-app')
        self.assertEqual(owner, 'myuser')
        self.assertEqual(name, 'my-app')

    def test_empty_studio_id_raises(self):
        with self.assertRaises(InvalidParameter):
            HubApi._parse_studio_id('')

    def test_no_slash_raises(self):
        with self.assertRaises(InvalidParameter):
            HubApi._parse_studio_id('no-slash-here')

    def test_multiple_slashes_raises(self):
        with self.assertRaises(InvalidParameter):
            HubApi._parse_studio_id('a/b/c')

    def test_empty_parts_raises(self):
        with self.assertRaises(InvalidParameter):
            HubApi._parse_studio_id('/name')
        with self.assertRaises(InvalidParameter):
            HubApi._parse_studio_id('owner/')


class TestStudioApiMocked(unittest.TestCase):
    """Cover every Studio HubApi method with mocked HTTP traffic."""

    def setUp(self):
        self.api = HubApi()
        self.api.endpoint = 'https://modelscope.cn'
        config = get_test_config()
        self.owner = config['owner']
        # Mock tests use a fixed test name to keep assertions deterministic.
        self.name = 'mock-studio'
        self.studio_id = f'{self.owner}/{self.name}'
        self.visibility = config['visibility']

    # ------------------------------------------------------------------
    # create_repo studio branch
    # ------------------------------------------------------------------
    @patch.object(HubApi, '_build_bearer_headers')
    @patch.object(HubApi, 'repo_exists', return_value=False)
    def test_create_repo_studio(self, mock_exists, mock_headers):
        mock_headers.return_value = {'Authorization': 'Bearer test-token'}
        mock_resp = _make_response(
            200, {
                'success': True,
                'data': {
                    'id': self.studio_id,
                    'repo_name': self.name,
                    'owner': self.owner,
                }
            })
        # Map configured visibility string -> Visibility enum value.
        visibility_value = (
            Visibility.PRIVATE
            if self.visibility == 'private' else Visibility.PUBLIC)
        expected_private = self.visibility == 'private'
        with patch.object(
                self.api.session, 'post', return_value=mock_resp) as mock_post:
            url = self.api.create_repo(
                self.studio_id,
                token='test-token',
                repo_type=REPO_TYPE_STUDIO,
                visibility=visibility_value,
                sdk_type=StudioSDKType.GRADIO,
                sdk_version='6.2.0',
                hardware=StudioHardware.DEFAULT,
            )
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            # First positional arg is the URL.
            url_arg = args[0] if args else kwargs.get('url', '')
            self.assertIn('/openapi/v1/studios', url_arg)
            # Body filters None values and contains studio-specific fields.
            body = kwargs.get('json', {})
            self.assertEqual(body.get('owner'), self.owner)
            self.assertEqual(body.get('repo_name'), self.name)
            self.assertEqual(body.get('sdk_type'), StudioSDKType.GRADIO)
            self.assertEqual(body.get('sdk_version'), '6.2.0')
            self.assertEqual(body.get('hardware'), StudioHardware.DEFAULT)
            # Visibility translates to ``private`` boolean; the filter keeps
            # explicit False values because they are not None.
            self.assertEqual(body.get('private'), expected_private)
            self.assertIn('Authorization', kwargs.get('headers', {}))
            # Returned URL points at the studios path.
            self.assertEqual(
                url, f'https://modelscope.cn/studios/{self.studio_id}')

    # ------------------------------------------------------------------
    # repo_exists studio branch
    # ------------------------------------------------------------------
    @patch.object(HubApi, '_build_bearer_headers')
    def test_repo_exists_studio_true(self, mock_headers):
        mock_headers.return_value = {'Authorization': 'Bearer test-token'}
        mock_resp = _make_response(200, {'success': True, 'data': {}})
        with patch.object(
                self.api.session, 'get', return_value=mock_resp) as mock_get:
            result = self.api.repo_exists(
                self.studio_id, repo_type=REPO_TYPE_STUDIO)
            self.assertTrue(result)
            args, kwargs = mock_get.call_args
            url_arg = args[0] if args else kwargs.get('url', '')
            self.assertIn(f'/openapi/v1/studios/{self.studio_id}', url_arg)

    @patch.object(HubApi, '_build_bearer_headers')
    def test_repo_exists_studio_false(self, mock_headers):
        mock_headers.return_value = {'Authorization': 'Bearer test-token'}
        mock_resp = _make_response(404, {})
        with patch.object(self.api.session, 'get', return_value=mock_resp):
            result = self.api.repo_exists(
                self.studio_id, repo_type=REPO_TYPE_STUDIO)
            self.assertFalse(result)

    # ------------------------------------------------------------------
    # deploy_studio
    # ------------------------------------------------------------------
    @patch.object(HubApi, '_build_bearer_headers')
    def test_deploy_studio(self, mock_headers):
        mock_headers.return_value = {'Authorization': 'Bearer test-token'}
        mock_resp = _make_response(
            200, {
                'success': True,
                'data': {
                    'status': StudioStatus.DEPLOYING,
                    'active_config': {
                        'hardware': StudioHardware.DEFAULT,
                        'sdk_type': StudioSDKType.GRADIO,
                        'sdk_version': '6.2.0',
                    },
                },
            })
        with patch.object(
                self.api.session, 'post', return_value=mock_resp) as mock_post:
            result = self.api.deploy_studio(self.studio_id)
            self.assertEqual(result['status'], StudioStatus.DEPLOYING)
            self.assertEqual(result['active_config']['sdk_type'],
                             StudioSDKType.GRADIO)
            args, kwargs = mock_post.call_args
            url_arg = args[0] if args else kwargs.get('url', '')
            self.assertEqual(
                url_arg, f'https://modelscope.cn/openapi/v1/studios/'
                f'{self.studio_id}/deploy')
            self.assertIn('Authorization', kwargs.get('headers', {}))

    # ------------------------------------------------------------------
    # stop_studio
    # ------------------------------------------------------------------
    @patch.object(HubApi, '_build_bearer_headers')
    def test_stop_studio(self, mock_headers):
        mock_headers.return_value = {'Authorization': 'Bearer test-token'}
        mock_resp = _make_response(200, {
            'success': True,
            'data': {
                'status': StudioStatus.STOPPING
            },
        })
        with patch.object(
                self.api.session, 'post', return_value=mock_resp) as mock_post:
            result = self.api.stop_studio(self.studio_id)
            self.assertEqual(result['status'], StudioStatus.STOPPING)
            args, kwargs = mock_post.call_args
            url_arg = args[0] if args else kwargs.get('url', '')
            self.assertEqual(
                url_arg, f'https://modelscope.cn/openapi/v1/studios/'
                f'{self.studio_id}/stop')

    # ------------------------------------------------------------------
    # get_studio_logs
    # ------------------------------------------------------------------
    @patch.object(HubApi, '_build_bearer_headers')
    def test_get_studio_logs_runtime(self, mock_headers):
        mock_headers.return_value = {'Authorization': 'Bearer test-token'}
        mock_resp = _make_response(
            200, {
                'success': True,
                'data': {
                    'logs': [
                        '2025-01-01 INFO Starting...',
                        '2025-01-01 INFO Ready',
                    ],
                    'page_num':
                    1,
                    'page_size':
                    100,
                    'total_count':
                    2,
                    'total_page_num':
                    1,
                },
            })
        with patch.object(
                self.api.session, 'get', return_value=mock_resp) as mock_get:
            result = self.api.get_studio_logs(
                self.studio_id, log_type='runtime')
            self.assertEqual(len(result['logs']), 2)
            args, kwargs = mock_get.call_args
            url_arg = args[0] if args else kwargs.get('url', '')
            self.assertEqual(
                url_arg, f'https://modelscope.cn/openapi/v1/studios/'
                f'{self.studio_id}/logs/runtime')
            params = kwargs.get('params', {})
            self.assertEqual(params['page_num'], 1)
            self.assertEqual(params['page_size'], 100)
            # Optional filters are omitted when not provided.
            self.assertNotIn('keyword', params)
            self.assertNotIn('start_timestamp', params)
            self.assertNotIn('end_timestamp', params)

    @patch.object(HubApi, '_build_bearer_headers')
    def test_get_studio_logs_build_with_filters(self, mock_headers):
        mock_headers.return_value = {'Authorization': 'Bearer test-token'}
        mock_resp = _make_response(200, {
            'success': True,
            'data': {
                'logs': ['error line'],
                'total_count': 1
            },
        })
        with patch.object(
                self.api.session, 'get', return_value=mock_resp) as mock_get:
            self.api.get_studio_logs(
                self.studio_id,
                log_type='build',
                keyword='error',
                page_num=2,
                page_size=50,
                start_timestamp=1000000,
                end_timestamp=2000000,
            )
            args, kwargs = mock_get.call_args
            url_arg = args[0] if args else kwargs.get('url', '')
            self.assertIn('/logs/build', url_arg)
            params = kwargs.get('params', {})
            self.assertEqual(params['keyword'], 'error')
            self.assertEqual(params['page_num'], 2)
            self.assertEqual(params['page_size'], 50)
            self.assertEqual(params['start_timestamp'], 1000000)
            self.assertEqual(params['end_timestamp'], 2000000)

    # ------------------------------------------------------------------
    # update_studio_settings
    # ------------------------------------------------------------------
    @patch.object(HubApi, '_build_bearer_headers')
    def test_update_studio_settings(self, mock_headers):
        mock_headers.return_value = {'Authorization': 'Bearer test-token'}
        mock_resp = _make_response(
            200, {
                'success': True,
                'data': {
                    'display_name': 'New Name',
                    'hardware': StudioHardware.GPU_16G,
                },
            })
        with patch.object(
                self.api.session, 'patch',
                return_value=mock_resp) as mock_patch:
            result = self.api.update_studio_settings(
                self.studio_id,
                display_name='New Name',
                hardware=StudioHardware.GPU_16G,
            )
            self.assertEqual(result['display_name'], 'New Name')
            mock_patch.assert_called_once()
            args, kwargs = mock_patch.call_args
            url_arg = args[0] if args else kwargs.get('url', '')
            self.assertEqual(
                url_arg, f'https://modelscope.cn/openapi/v1/studios/'
                f'{self.studio_id}/settings')
            body = kwargs.get('json', {})
            self.assertEqual(body['display_name'], 'New Name')
            self.assertEqual(body['hardware'], StudioHardware.GPU_16G)

    @patch.object(HubApi, '_build_bearer_headers')
    def test_update_studio_settings_filters_none(self, mock_headers):
        mock_headers.return_value = {'Authorization': 'Bearer test-token'}
        mock_resp = _make_response(200, {'success': True, 'data': {}})
        with patch.object(
                self.api.session, 'patch',
                return_value=mock_resp) as mock_patch:
            self.api.update_studio_settings(
                self.studio_id,
                display_name='Name',
                sdk_type=None,
                hardware=None,
            )
            args, kwargs = mock_patch.call_args
            body = kwargs.get('json', {})
            self.assertIn('display_name', body)
            self.assertNotIn('sdk_type', body)
            self.assertNotIn('hardware', body)

    @patch.object(HubApi, '_build_bearer_headers')
    def test_update_studio_settings_visibility_false(self, mock_headers):
        """Explicit ``private=False`` must be forwarded (False != None)."""
        mock_headers.return_value = {'Authorization': 'Bearer test-token'}
        mock_resp = _make_response(200, {'success': True, 'data': {}})
        with patch.object(
                self.api.session, 'patch',
                return_value=mock_resp) as mock_patch:
            self.api.update_studio_settings(self.studio_id, private=False)
            args, kwargs = mock_patch.call_args
            body = kwargs.get('json', {})
            self.assertIn('private', body)
            self.assertEqual(body['private'], False)

    # ------------------------------------------------------------------
    # secrets CRUD
    # ------------------------------------------------------------------
    @patch.object(HubApi, '_build_bearer_headers')
    def test_list_studio_secrets(self, mock_headers):
        mock_headers.return_value = {'Authorization': 'Bearer test-token'}
        mock_resp = _make_response(
            200, {
                'success': True,
                'data': {
                    'secrets': [{
                        'key': 'API_KEY'
                    }, {
                        'key': 'DB_URL'
                    }],
                },
            })
        with patch.object(
                self.api.session, 'get', return_value=mock_resp) as mock_get:
            result = self.api.list_studio_secrets(self.studio_id)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]['key'], 'API_KEY')
            args, kwargs = mock_get.call_args
            url_arg = args[0] if args else kwargs.get('url', '')
            self.assertEqual(
                url_arg, f'https://modelscope.cn/openapi/v1/studios/'
                f'{self.studio_id}/secrets')

    @patch.object(HubApi, '_build_bearer_headers')
    def test_list_studio_secrets_empty(self, mock_headers):
        mock_headers.return_value = {'Authorization': 'Bearer test-token'}
        mock_resp = _make_response(200, {'success': True, 'data': {}})
        with patch.object(self.api.session, 'get', return_value=mock_resp):
            result = self.api.list_studio_secrets(self.studio_id)
            self.assertEqual(result, [])

    @patch.object(HubApi, '_build_bearer_headers')
    def test_add_studio_secret(self, mock_headers):
        mock_headers.return_value = {'Authorization': 'Bearer test-token'}
        mock_resp = _make_response(200, {'success': True})
        with patch.object(
                self.api.session, 'post', return_value=mock_resp) as mock_post:
            self.api.add_studio_secret(self.studio_id, 'MY_KEY', 'my_value')
            args, kwargs = mock_post.call_args
            url_arg = args[0] if args else kwargs.get('url', '')
            self.assertEqual(
                url_arg, f'https://modelscope.cn/openapi/v1/studios/'
                f'{self.studio_id}/secrets')
            body = kwargs.get('json', {})
            self.assertEqual(body, {'key': 'MY_KEY', 'value': 'my_value'})

    @patch.object(HubApi, '_build_bearer_headers')
    def test_update_studio_secret(self, mock_headers):
        mock_headers.return_value = {'Authorization': 'Bearer test-token'}
        mock_resp = _make_response(200, {'success': True})
        with patch.object(
                self.api.session, 'put', return_value=mock_resp) as mock_put:
            self.api.update_studio_secret(self.studio_id, 'MY_KEY',
                                          'new_value')
            mock_put.assert_called_once()
            args, kwargs = mock_put.call_args
            url_arg = args[0] if args else kwargs.get('url', '')
            self.assertEqual(
                url_arg, f'https://modelscope.cn/openapi/v1/studios/'
                f'{self.studio_id}/secrets')
            body = kwargs.get('json', {})
            self.assertEqual(body, {'key': 'MY_KEY', 'value': 'new_value'})

    @patch.object(HubApi, '_build_bearer_headers')
    def test_delete_studio_secret(self, mock_headers):
        mock_headers.return_value = {'Authorization': 'Bearer test-token'}
        mock_resp = _make_response(200, {'success': True})
        with patch.object(
                self.api.session, 'delete',
                return_value=mock_resp) as mock_delete:
            self.api.delete_studio_secret(self.studio_id, 'MY_KEY')
            args, kwargs = mock_delete.call_args
            url_arg = args[0] if args else kwargs.get('url', '')
            self.assertEqual(
                url_arg, f'https://modelscope.cn/openapi/v1/studios/'
                f'{self.studio_id}/secrets')
            body = kwargs.get('json', {})
            self.assertEqual(body, {'key': 'MY_KEY'})

    # ------------------------------------------------------------------
    # Token / endpoint plumbing
    # ------------------------------------------------------------------
    @patch.object(HubApi, '_build_bearer_headers')
    def test_methods_pass_token_to_headers(self, mock_headers):
        """Every studio method should propagate ``token`` to the bearer header
        builder so that explicit CLI tokens are honoured."""
        mock_headers.return_value = {'Authorization': 'Bearer custom'}
        mock_resp = _make_response(200, {'success': True, 'data': {}})
        with patch.object(self.api.session, 'post', return_value=mock_resp):
            self.api.deploy_studio(self.studio_id, token='custom')
            self.api.stop_studio(self.studio_id, token='custom')
        with patch.object(self.api.session, 'get', return_value=mock_resp):
            self.api.get_studio_logs(self.studio_id, token='custom')
            self.api.list_studio_secrets(self.studio_id, token='custom')
        with patch.object(self.api.session, 'patch', return_value=mock_resp):
            self.api.update_studio_settings(
                self.studio_id, display_name='X', token='custom')
        # Each call requested ``token_required=True`` with token='custom'.
        for call in mock_headers.call_args_list:
            self.assertEqual(call.kwargs.get('token'), 'custom')
            self.assertTrue(call.kwargs.get('token_required'))


if __name__ == '__main__':
    unittest.main()
