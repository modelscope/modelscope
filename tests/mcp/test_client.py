# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio
import unittest
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from mcp import ClientSession
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED, Implementation, Tool

from modelscope.mcp.client import (MCPClient, MCPClientError,
                                   MCPConnectionError, MCPTimeoutError,
                                   MCPToolExecutionError)


class TestMCPClientAsync(unittest.IsolatedAsyncioTestCase):
    """Async tests for MCPClient core functionality"""

    def setUp(self):
        # Basic configurations
        self.stdio_config = {
            'type': 'stdio',
            'command': ['python', '-m', 'test_server']
        }
        self.sse_config = {
            'type': 'sse',
            'url': 'https://api.example.com/mcp/sse'
        }
        self.http_config = {
            'type': 'streamable_http',
            'url': 'https://api.example.com/mcp/http'
        }

        # Mock tool data
        self.mock_tools = [
            Tool(name='tool1', description='Test tool 1', inputSchema={}),
            Tool(name='tool2', description='Test tool 2', inputSchema={})
        ]

        # Mock initialization result
        self.mock_init_result = MagicMock()
        self.mock_init_result.serverInfo = MagicMock()
        self.mock_init_result.serverInfo.name = 'test-server'
        self.mock_init_result.serverInfo.version = '1.0.0'

    async def test_connect_success_stdio(self):
        """Test successful STDIO connection"""
        with patch('modelscope.mcp.client.stdio_client') as mock_stdio:
            # Mock stdio_client returning read/write streams
            mock_read = AsyncMock()
            mock_write = AsyncMock()
            mock_stdio.return_value.__aenter__.return_value = (mock_read,
                                                               mock_write)

            # Mock ClientSession initialization
            with patch(
                    'modelscope.mcp.client.ClientSession') as mock_session_cls:
                mock_session = AsyncMock(spec=ClientSession)
                mock_session.initialize.return_value = self.mock_init_result
                mock_session_cls.return_value.__aenter__.return_value = mock_session

                client = MCPClient(self.stdio_config)
                await client.connect()

                # Verify connection state
                self.assertTrue(client.is_connected())
                self.assertEqual(client.get_server_name(), 'test-server')
                self.assertEqual(client.get_server_info(), {
                    'name': 'test-server',
                    'version': '1.0.0'
                })

                # Verify underlying calls
                mock_stdio.assert_called_once()
                mock_session.initialize.assert_awaited_once()

    async def test_connect_failure(self):
        """Test connection failure scenario"""
        with patch('modelscope.mcp.client.sse_client') as mock_sse:
            # Mock connection throwing exception
            mock_sse.return_value.__aenter__.side_effect = Exception(
                'Connection refused')

            client = MCPClient(self.sse_config)
            with self.assertRaises(MCPConnectionError) as context:
                await client.connect()

            self.assertIn('Connection failed: Connection refused',
                          str(context.exception))
            self.assertFalse(client.is_connected())

    async def test_call_tool_success(self):
        """Test successful tool call"""
        with patch('modelscope.mcp.client.streamablehttp_client') as mock_http:
            # Mock HTTP client and session
            mock_read = AsyncMock()
            mock_write = AsyncMock()
            mock_http.return_value.__aenter__.return_value = (mock_read,
                                                              mock_write)

            with patch(
                    'modelscope.mcp.client.ClientSession') as mock_session_cls:
                mock_session = AsyncMock(spec=ClientSession)
                mock_session.initialize.return_value = self.mock_init_result

                # Mock tool call result
                mock_result = MagicMock()
                mock_result.content = [
                    MagicMock(type='text', text='First part'),
                    MagicMock(type='text', text='Second part')
                ]
                mock_session.call_tool.return_value = mock_result

                mock_session_cls.return_value.__aenter__.return_value = mock_session

                client = MCPClient(self.http_config)
                await client.connect()

                # Execute tool call
                result = await client.call_tool('tool1', {'param': 'value'})

                # Verify result
                self.assertEqual(result, 'First part\n\nSecond part')
                mock_session.call_tool.assert_awaited_once_with(
                    'tool1', {'param': 'value'},
                    read_timeout_seconds=client.read_timeout)

    async def test_call_tool_not_connected(self):
        """Test calling tool when not connected"""
        client = MCPClient(self.sse_config)
        with self.assertRaises(MCPConnectionError) as context:
            await client.call_tool('tool1', {})

        self.assertIn('Not connected to server', str(context.exception))

    async def test_call_tool_mcp_error_connection_closed(self):
        """Test connection closed error during tool call"""
        with patch('modelscope.mcp.client.stdio_client') as mock_stdio:
            mock_stdio.return_value.__aenter__.return_value = (AsyncMock(),
                                                               AsyncMock())

            with patch(
                    'modelscope.mcp.client.ClientSession') as mock_session_cls:
                mock_session = AsyncMock(spec=ClientSession)
                mock_session.initialize.return_value = self.mock_init_result

                # Mock connection closed error
                mock_error = MagicMock(
                    code=CONNECTION_CLOSED, message='Connection lost')
                mcp_error = McpError(mock_error)
                mock_session.call_tool.side_effect = mcp_error

                mock_session_cls.return_value.__aenter__.return_value = mock_session

                client = MCPClient(self.stdio_config)
                await client.connect()

                with self.assertRaises(MCPConnectionError) as context:
                    await client.call_tool('tool1', {})

                self.assertIn('Connection lost while calling tool',
                              str(context.exception))
                self.assertFalse(client.is_connected(
                ))  # Connection should be marked as disconnected

    async def test_call_tool_timeout(self):
        """Test tool call timeout"""
        with patch('modelscope.mcp.client.sse_client') as mock_sse:
            mock_sse.return_value.__aenter__.return_value = (AsyncMock(),
                                                             AsyncMock())

            with patch(
                    'modelscope.mcp.client.ClientSession') as mock_session_cls:
                mock_session = AsyncMock(spec=ClientSession)
                mock_session.initialize.return_value = self.mock_init_result
                mock_session.call_tool.side_effect = asyncio.TimeoutError()

                mock_session_cls.return_value.__aenter__.return_value = mock_session

                client = MCPClient(
                    self.sse_config, timeout=timedelta(seconds=10))
                await client.connect()

                with self.assertRaises(MCPTimeoutError) as context:
                    await client.call_tool('tool1', {})

                self.assertIn('timed out after 0:00:10',
                              str(context.exception))

    async def test_list_tools_success(self):
        """Test successful tool listing"""
        with patch('modelscope.mcp.client.streamablehttp_client') as mock_http:
            mock_http.return_value.__aenter__.return_value = (AsyncMock(),
                                                              AsyncMock())

            with patch(
                    'modelscope.mcp.client.ClientSession') as mock_session_cls:
                mock_session = AsyncMock(spec=ClientSession)
                mock_session.initialize.return_value = self.mock_init_result

                # Mock tool list result
                mock_list_result = MagicMock()
                mock_list_result.tools = self.mock_tools
                mock_session.list_tools.return_value = mock_list_result

                mock_session_cls.return_value.__aenter__.return_value = mock_session

                client = MCPClient(self.http_config)
                await client.connect()

                tools = await client.list_tools()

                self.assertEqual(len(tools), 2)
                self.assertEqual(tools[0].name, 'tool1')
                mock_session.list_tools.assert_awaited_once()

    async def test_list_tools_not_connected(self):
        """Test listing tools when not connected"""
        client = MCPClient(self.stdio_config)
        with self.assertRaises(MCPConnectionError) as context:
            await client.list_tools()

        self.assertIn('Not connected to server', str(context.exception))

    async def test_context_manager(self):
        """Test async context manager"""
        with patch('modelscope.mcp.client.sse_client') as mock_sse:
            mock_sse.return_value.__aenter__.return_value = (AsyncMock(),
                                                             AsyncMock())

            with patch(
                    'modelscope.mcp.client.ClientSession') as mock_session_cls:
                mock_session = AsyncMock(spec=ClientSession)
                mock_session.initialize.return_value = self.mock_init_result
                mock_session.list_tools.return_value = MagicMock(
                    tools=self.mock_tools)
                mock_session_cls.return_value.__aenter__.return_value = mock_session

                async with MCPClient(self.sse_config) as client:
                    self.assertTrue(client.is_connected())
                    tools = await client.list_tools()
                    self.assertEqual(len(tools), 2)

                # Should be disconnected after exiting context
                self.assertFalse(client.is_connected())

    async def test_disconnect(self):
        """Test manual disconnection"""
        with patch('modelscope.mcp.client.stdio_client') as mock_stdio:
            mock_stdio.return_value.__aenter__.return_value = (AsyncMock(),
                                                               AsyncMock())

            with patch(
                    'modelscope.mcp.client.ClientSession') as mock_session_cls:
                mock_session = AsyncMock(spec=ClientSession)
                mock_session.initialize.return_value = self.mock_init_result
                mock_session_cls.return_value.__aenter__.return_value = mock_session

                client = MCPClient(self.stdio_config)
                await client.connect()
                self.assertTrue(client.is_connected())

                await client.disconnect()
                self.assertFalse(client.is_connected())
                self.assertIsNone(client.session)


if __name__ == '__main__':
    unittest.main()
