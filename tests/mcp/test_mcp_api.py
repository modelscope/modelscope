# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.mcp.api import MCPApi
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import TEST_ACCESS_TOKEN1, test_level

logger = get_logger()


class MCPApiTest(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.api = MCPApi()
        self.api.login(TEST_ACCESS_TOKEN1)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_list_mcp_servers(self):
        """Test list_mcp_servers functionality and validation."""
        # Test parameter validation
        with self.assertRaises(ValueError):
            self.api.list_mcp_servers(page_number=0)
        with self.assertRaises(ValueError):
            self.api.list_mcp_servers(page_size=0)

        # Test basic functionality
        result = self.api.list_mcp_servers(page_size=5)

        # Verify response structure and content
        self.assertIn('total_counts', result)
        self.assertIn('servers', result)
        self.assertGreater(result['total_counts'], 0)
        self.assertGreater(len(result['servers']), 0)

        # Verify server structure
        server = result['servers'][0]
        for field in ['name', 'id', 'description']:
            self.assertIn(field, server)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_list_operational_mcp_servers(self):
        """Test list_operational_mcp_servers functionality."""
        result = self.api.list_operational_mcp_servers()

        # Verify response structure
        for field in ['total_counts', 'servers', 'mcpServers']:
            self.assertIn(field, result)

        # Verify mcpServers configuration if exists
        if result['mcpServers']:
            first_config = list(result['mcpServers'].values())[0]
            self.assertEqual(first_config['type'], 'sse')
            self.assertTrue(first_config['url'].startswith('https://'))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_get_mcp_server(self):
        """Test get_mcp_server functionality and validation."""
        # Test parameter validation
        with self.assertRaises(ValueError):
            self.api.get_mcp_server('')
        with self.assertRaises(ValueError):
            self.api.get_mcp_server(None)

        # Test with real server
        result = self.api.get_mcp_server('@modelcontextprotocol/fetch')

        # Verify response structure
        for field in ['name', 'id', 'description', 'service_config']:
            self.assertIn(field, result)
        self.assertEqual(result['id'], '@modelcontextprotocol/fetch')


if __name__ == '__main__':
    unittest.main()
