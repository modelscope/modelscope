# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest
from datetime import timedelta

from modelscope.mcp.client import (MCPClient, MCPClientError,
                                   MCPConnectionError, MCPTimeoutError,
                                   MCPToolExecutionError)
from modelscope.utils.logger import get_logger

logger = get_logger()


class ClientUtilsTest(unittest.TestCase):

    def test_mcp_client_initialization_stdio(self):
        """Test MCPClient initialization with STDIO configuration."""
        config = {'type': 'stdio', 'command': ['python', '-m', 'test_server']}
        client = MCPClient(config)

        self.assertEqual(client.get_transport_type(), 'stdio')
        self.assertEqual(client.get_server_name(), 'stdio-python')
        self.assertFalse(client.is_connected())

    def test_mcp_client_initialization_sse(self):
        """Test MCPClient initialization with SSE configuration."""
        config = {'type': 'sse', 'url': 'https://api.example.com/mcp/sse'}
        client = MCPClient(config)

        self.assertEqual(client.get_transport_type(), 'sse')
        self.assertEqual(client.get_server_name(), 'sse-api')
        self.assertFalse(client.is_connected())

    def test_mcp_client_initialization_streamable_http(self):
        """Test MCPClient initialization with streamable HTTP configuration."""
        config = {
            'type': 'streamable_http',
            'url': 'https://test.modelscope.cn/api/mcp'
        }
        client = MCPClient(config)

        self.assertEqual(client.get_transport_type(), 'streamable_http')
        self.assertEqual(client.get_server_name(), 'streamable_http-test')

    def test_mcp_client_initialization_with_timeout(self):
        """Test MCPClient initialization with custom timeout."""
        config = {'type': 'sse', 'url': 'https://api.example.com/sse'}
        timeout = timedelta(seconds=60)

        client = MCPClient(config, timeout=timeout)
        self.assertEqual(client.read_timeout, timeout)

    def test_mcp_client_invalid_config_empty(self):
        """Test MCPClient with empty configuration."""
        with self.assertRaises(ValueError) as context:
            MCPClient({})
        self.assertIn('MCP server configuration is required',
                      str(context.exception))

    def test_mcp_client_invalid_config_no_type(self):
        """Test MCPClient with missing type."""
        config = {'url': 'https://api.example.com'}
        with self.assertRaises(ValueError) as context:
            MCPClient(config)
        self.assertIn('Server type is required', str(context.exception))

    def test_mcp_client_invalid_config_no_url_or_command(self):
        """Test MCPClient with missing URL and command."""
        config = {'type': 'sse'}
        with self.assertRaises(ValueError) as context:
            MCPClient(config)
        self.assertIn('Server URL or command is required',
                      str(context.exception))

    def test_mcp_client_invalid_transport_type(self):
        """Test MCPClient with unsupported transport type."""
        config = {'type': 'unsupported', 'url': 'https://api.example.com'}
        with self.assertRaises(ValueError) as context:
            MCPClient(config)
        self.assertIn('Unsupported transport type: unsupported',
                      str(context.exception))

    def test_generate_server_name_stdio_list_command(self):
        """Test server name generation for STDIO with list command."""
        config = {
            'type': 'stdio',
            'command': ['python', '-m', 'weather_server']
        }
        client = MCPClient(config)
        self.assertEqual(client.get_server_name(), 'stdio-python')

    def test_generate_server_name_stdio_string_command(self):
        """Test server name generation for STDIO with string command."""
        config = {'type': 'stdio', 'command': 'weather-tool'}
        client = MCPClient(config)
        self.assertEqual(client.get_server_name(), 'stdio-weather-tool')

    def test_generate_server_name_sse_complex_url(self):
        """Test server name generation for SSE with complex URL."""
        config = {
            'type': 'sse',
            'url': 'https://mcp.api-inference.modelscope.net/weather/sse'
        }
        client = MCPClient(config)
        self.assertEqual(client.get_server_name(), 'sse-mcp')

    def test_generate_server_name_default_fallback(self):
        """Test server name generation fallback for unknown type."""
        # Temporarily modify config after validation
        config = {'type': 'sse', 'url': 'https://example.com'}
        client = MCPClient(config)

        # Simulate invalid URL parsing
        client.mcp_server = {'type': 'unknown'}
        name = client._generate_server_name()
        self.assertEqual(name, 'mcp-unknown-server')

    def test_validate_config_nested_mcpservers(self):
        """Test configuration validation with nested mcpServers structure."""
        config = {
            'mcpServers': {
                'weather': {
                    'type': 'sse',
                    'url': 'https://api.example.com/weather'
                }
            }
        }
        client = MCPClient(config)

        # Should extract the nested server config
        self.assertEqual(client.mcp_server['type'], 'sse')
        self.assertEqual(client.mcp_server['url'],
                         'https://api.example.com/weather')

    def test_validate_config_nested_empty_servers(self):
        """Test configuration validation with empty mcpServers."""
        config = {'mcpServers': {}}
        with self.assertRaises(ValueError) as context:
            MCPClient(config)
        self.assertIn('No servers found in mcpServers', str(context.exception))

    def test_validate_config_nested_invalid_server_config(self):
        """Test configuration validation with invalid nested server config."""
        config = {
            'mcpServers': {
                'weather': 'invalid_config'  # Should be dict
            }
        }
        with self.assertRaises(ValueError) as context:
            MCPClient(config)
        self.assertIn('must be a dictionary', str(context.exception))

    def test_server_info_initial_state(self):
        """Test initial server info state."""
        config = {'type': 'sse', 'url': 'https://api.example.com'}
        client = MCPClient(config)

        self.assertIsNone(client.get_server_info())

    def test_exception_hierarchy(self):
        """Test exception class hierarchy."""
        self.assertTrue(issubclass(MCPConnectionError, MCPClientError))
        self.assertTrue(issubclass(MCPToolExecutionError, MCPClientError))
        self.assertTrue(issubclass(MCPTimeoutError, MCPClientError))
        self.assertTrue(issubclass(MCPClientError, Exception))


if __name__ == '__main__':
    unittest.main()
