# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest
from types import SimpleNamespace
from unittest import mock

from modelscope.hub.mcp_api import MCPApi, MCPApiRequestError
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import TEST_ACCESS_TOKEN1, test_level

logger = get_logger()


class MCPApiTest(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.api = MCPApi()
        self.api.login(TEST_ACCESS_TOKEN1)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_list_mcp_servers(self):
        """Test list_mcp_servers functionality and validation."""
        result = self.api.list_mcp_servers(total_count=5)

        # Verify response structure and content
        self.assertIn('total_count', result)
        self.assertIn('servers', result)
        self.assertGreater(result['total_count'], 0)
        self.assertGreater(len(result['servers']), 0)

        # Verify server structure
        server = result['servers'][0]
        for field in ['name', 'id', 'description']:
            self.assertIn(field, server)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_list_operational_mcp_servers(self):
        """Test list_operational_mcp_servers functionality."""
        result = self.api.list_operational_mcp_servers()

        # Verify response structure - corrected field names
        for field in ['total_count', 'servers']:
            self.assertIn(field, result)

        # Verify servers structure if exists
        if result['servers']:
            first_server = result['servers'][0]
            for field in ['name', 'id', 'description', 'mcp_servers']:
                self.assertIn(field, first_server)

            # Verify mcp_servers configuration if exists
            if first_server['mcp_servers']:
                first_config = first_server['mcp_servers'][0]
                self.assertIn('type', first_config)
                self.assertIn('url', first_config)
                self.assertTrue(first_config['url'].startswith('https://'))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_get_mcp_server(self):
        """Test get_mcp_server functionality and validation."""
        result = self.api.get_mcp_server('@modelcontextprotocol/fetch')

        # Verify response structure
        for field in ['name', 'id', 'description', 'servers']:
            self.assertIn(field, result)
        self.assertEqual(result['id'], '@modelcontextprotocol/fetch')


class MCPApiCompatTest(unittest.TestCase):

    def test_list_mcp_servers_delegates_to_modelscope_hub(self):
        api = MCPApi()
        delegated_api = mock.MagicMock()
        delegated_api.list_mcp_servers.return_value = SimpleNamespace(
            items=[{
                'Name': 'Fetch',
                'Id': '@modelcontextprotocol/fetch',
                'Description': 'Fetch pages'
            }],
            total_count=1,
        )

        with mock.patch(
                'modelscope_hub.api.HubApi',
                return_value=delegated_api) as hub_api_cls:
            result = api.list_mcp_servers(
                token='ms-readonly',
                filter={'category': 'tools'},
                total_count=2,
                search='fetch')

        hub_api_cls.assert_called_once_with(
            token='ms-readonly', endpoint=api.endpoint)
        delegated_api.list_mcp_servers.assert_called_once_with(
            search='fetch',
            page_number=1,
            page_size=2,
            filter={'category': 'tools'},
        )
        self.assertEqual(result['total_count'], 1)
        self.assertEqual(result['servers'][0]['id'],
                         '@modelcontextprotocol/fetch')

    def test_list_operational_mcp_servers_delegates_to_modelscope_hub(self):
        api = MCPApi()
        delegated_api = mock.MagicMock()
        delegated_api.list_operational_mcp_servers.return_value = SimpleNamespace(
            items=[{
                'name':
                'Fetch',
                'id':
                '@modelcontextprotocol/fetch',
                'description':
                'Fetch pages',
                'operational_urls': [{
                    'url':
                    'https://mcp.api-inference.modelscope.net/abc/sse'
                }, {
                    'url':
                    'https://mcp.api-inference.modelscope.net/abc/streamable_http'
                }],
            }],
            total_count=1,
        )

        with mock.patch(
                'modelscope_hub.api.HubApi',
                return_value=delegated_api) as hub_api_cls:
            result = api.list_operational_mcp_servers(token='ms-readonly')

        hub_api_cls.assert_called_once_with(
            token='ms-readonly', endpoint=api.endpoint)
        delegated_api.list_operational_mcp_servers.assert_called_once_with()
        self.assertEqual(result['total_count'], 1)
        server = result['servers'][0]
        self.assertEqual(server['id'], '@modelcontextprotocol/fetch')
        # The transport is the final path segment of the hosted URL.
        self.assertEqual([c['type'] for c in server['mcp_servers']],
                         ['sse', 'streamable_http'])

    def test_get_mcp_server_delegates_to_modelscope_hub(self):
        api = MCPApi()
        delegated_api = mock.MagicMock()
        delegated_api.get_mcp_server.return_value = {
            'name':
            'Fetch',
            'id':
            '@modelcontextprotocol/fetch',
            'description':
            'Fetch pages',
            'operational_urls': [{
                'url':
                'https://mcp.api-inference.modelscope.net/abc/sse'
            }],
        }

        with mock.patch(
                'modelscope_hub.api.HubApi',
                return_value=delegated_api) as hub_api_cls:
            result = api.get_mcp_server(
                '@modelcontextprotocol/fetch', token='ms-readonly')

        hub_api_cls.assert_called_once_with(
            token='ms-readonly', endpoint=api.endpoint)
        delegated_api.get_mcp_server.assert_called_once_with(
            '@modelcontextprotocol/fetch', get_operational_url=True)
        self.assertEqual(result['id'], '@modelcontextprotocol/fetch')
        self.assertEqual(
            result['servers'],
            [{
                'type': 'sse',
                'url': 'https://mcp.api-inference.modelscope.net/abc/sse'
            }])

    def test_deploy_mcp_server_delegates_and_drops_unset_options(self):
        api = MCPApi()
        delegated_api = mock.MagicMock()
        delegated_api.deploy_mcp_server.return_value = {'url': 'https://x/sse'}

        with mock.patch(
                'modelscope_hub.api.HubApi', return_value=delegated_api):
            result = api.deploy_mcp_server(
                '@modelcontextprotocol/fetch',
                transport_type='sse',
                env_info={'KEY': 'value'},
                token='ms-write')

        delegated_api.deploy_mcp_server.assert_called_once_with(
            '@modelcontextprotocol/fetch',
            payload={
                'transport_type': 'sse',
                'env_info': {
                    'KEY': 'value'
                }
            })
        self.assertEqual(result['url'], 'https://x/sse')

    def test_undeploy_mcp_server_delegates_to_modelscope_hub(self):
        api = MCPApi()
        delegated_api = mock.MagicMock()

        with mock.patch(
                'modelscope_hub.api.HubApi', return_value=delegated_api):
            api.undeploy_mcp_server(
                '@modelcontextprotocol/fetch', token='ms-write')

        delegated_api.undeploy_mcp_server.assert_called_once_with(
            '@modelcontextprotocol/fetch')

    def test_deploy_and_undeploy_reject_an_empty_server_id(self):
        api = MCPApi()
        with self.assertRaises(ValueError):
            api.deploy_mcp_server('')
        with self.assertRaises(ValueError):
            api.undeploy_mcp_server('')

    def test_delegation_failures_are_wrapped(self):
        api = MCPApi()
        delegated_api = mock.MagicMock()
        delegated_api.get_mcp_server.side_effect = RuntimeError('boom')

        with mock.patch(
                'modelscope_hub.api.HubApi', return_value=delegated_api):
            with self.assertRaises(MCPApiRequestError):
                api.get_mcp_server('@a/b', token='ms-any')


if __name__ == '__main__':
    unittest.main()
