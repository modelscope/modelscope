# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest
from types import SimpleNamespace
from unittest import mock

from modelscope.hub.mcp_api import MCPApi
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


if __name__ == '__main__':
    unittest.main()
