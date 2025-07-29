#!/usr/bin/env python3
"""
MCP Client - A concise implementation based on the official MCP Python SDK
"""

import asyncio
import time
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

# Import official MCP SDK
from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED, Implementation

from modelscope.utils.logger import get_logger

# Constants
DEFAULT_CLIENT_INFO = Implementation(
    name='ModelScope-mcp-client', version='1.0.0')

DEFAULT_READ_TIMEOUT = timedelta(seconds=30)
DEFAULT_HTTP_TIMEOUT = timedelta(seconds=30)
DEFAULT_SSE_READ_TIMEOUT = timedelta(seconds=30)

# Logger
logger = get_logger(__name__)


# Exception classes
class MCPClientError(Exception):
    """Base MCP client exception"""
    pass


class MCPConnectionError(MCPClientError):
    """MCP connection exception"""
    pass


class MCPToolExecutionError(MCPClientError):
    """MCP tool execution exception"""
    pass


class MCPTimeoutError(MCPClientError):
    """MCP timeout exception"""
    pass


class MCPClient:
    """
    MCP Client - A Python client for Model Context Protocol servers

    This client provides a simple, async interface to connect to MCP servers and execute tools.
    MCP (Model Context Protocol) allows AI models to securely access external data and services
    through standardized tool interfaces.

    Key Features:
    - Multiple transport types: STDIO, SSE, Streamable HTTP
    - Automatic connection management with context managers
    - Built-in error handling and timeout management
    - Tool discovery and execution
    - Concurrent tool execution support

    Supported Server Types:
    - STDIO: Local command-line tools and scripts
    - SSE: Server-Sent Events for real-time communication
    - HTTP: RESTful API endpoints with streaming support

    Basic Usage:

    Simple connection and tool execution:
    >>> import asyncio
    >>>
    >>> async def quick_example():
    ...     # Connect to a server
    ...     async with MCPClient({
    ...         "type": "sse",
    ...         "url": "https://api.example.com/mcp"
    ...     }) as client:
    ...
    ...         # Discover available tools
    ...         tools = await client.list_tools()
    ...         print(f"Found {len(tools)} tools")
    ...
    ...         # Execute a tool
    ...         result = await client.call_tool("search", {"query": "python"})
    ...         print(result)

    Configuration Examples:

    Local STDIO server:
    >>> client = MCPClient({
    ...     "type": "stdio",
    ...     "command": ["python", "-m", "my_mcp_server"]
    ... })

    Remote SSE server:
    >>> client = MCPClient({
    ...     "type": "sse",
    ...     "url": "https://api.example.com/mcp/sse"
    ... })

    HTTP streaming server:
    >>> client = MCPClient({
    ...     "type": "streamable_http",
    ...     "url": "https://api.example.com/mcp/http"
    ... })

    Context Manager Pattern (Recommended):
    >>> async def recommended_usage():
    ...     async with MCPClient(config) as client:
    ...         # Connection automatically managed
    ...         tools = await client.list_tools()
    ...
    ...         for tool in tools:
    ...             result = await client.call_tool(tool.name, {})
    ...             print(f"{tool.name}: {result}")
    ...     # Automatically disconnected here

    Manual Connection Management:
    >>> async def manual_management():
    ...     client = MCPClient(config)
    ...
    ...     try:
    ...         await client.connect()
    ...
    ...         if client.is_connected():
    ...             tools = await client.list_tools()
    ...             result = await client.call_tool("tool_name", {"param": "value"})
    ...     finally:
    ...         await client.disconnect()

    Error Handling:
    >>> async def safe_usage():
    ...     try:
    ...         async with MCPClient(config) as client:
    ...             result = await client.call_tool("risky_tool", {})
    ...     except MCPConnectionError:
    ...         print("Failed to connect to server")
    ...     except MCPToolExecutionError:
    ...         print("Tool execution failed")
    ...     except MCPTimeoutError:
    ...         print("Operation timed out")

    Concurrent Tool Execution:
    >>> async def concurrent_example():
    ...     async with MCPClient(config) as client:
    ...         # Execute multiple tools in parallel
    ...         tasks = [
    ...             client.call_tool("tool1", {"param": "value1"}),
    ...             client.call_tool("tool2", {"param": "value2"}),
    ...             client.call_tool("tool3", {"param": "value3"})
    ...         ]
    ...
    ...         results = await asyncio.gather(*tasks, return_exceptions=True)
    ...         for i, result in enumerate(results):
    ...             print(f"Tool {i+1}: {result}")
    """

    def __init__(self,
                 mcp_server: Dict[str, Any],
                 timeout: Optional[timedelta] = None):
        """
        Initialize MCP client

        Args:
            mcp_server: MCP server configuration
            default_timeout: Default timeout for tool calls (defaults to 30 seconds)
        """
        if not mcp_server:
            raise ValueError('MCP server configuration is required')

        self.mcp_server = mcp_server
        self.session: Optional[ClientSession] = None
        self.exit_stack: Optional[AsyncExitStack] = None
        self.client_info = DEFAULT_CLIENT_INFO
        self.connected = False
        self.read_timeout = timeout or DEFAULT_READ_TIMEOUT
        self.server_info: Optional[Dict[str, Any]] = None  # Server information

        # Validate configuration
        self._validate_config()

        # Auto-generate server name (may be updated after connection)
        self.server_name = self._generate_server_name()

    def _generate_server_name(self) -> str:
        """Auto-generate server name"""
        config = self.mcp_server

        # Extract meaningful name from configuration
        if 'type' in config:
            transport_type = config['type']

            if transport_type == 'stdio' and 'command' in config:
                # Extract name from command
                command = config['command']
                if isinstance(command, list) and command:
                    return f'stdio-{command[0]}'
                elif isinstance(command, str):
                    return f'stdio-{command}'

            elif transport_type in ['sse', 'streamable_http'
                                    ] and 'url' in config:
                # Extract domain from URL
                url = config['url']
                try:
                    parsed = urlparse(url)
                    domain = parsed.netloc.split('.')[
                        0]  # Get first domain part
                    return f'{transport_type}-{domain}'
                except Exception:
                    return f'{transport_type}-server'

        # Default name
        return f"mcp-{config.get('type', 'unknown')}-server"

    def _validate_config(self) -> None:
        """Validate MCP server configuration"""
        config = self.mcp_server

        # Check for mcpServers nested structure
        if 'mcpServers' in config:
            servers = config['mcpServers']
            if not servers:
                raise ValueError('No servers found in mcpServers')

            # Get first server configuration
            first_server_name = list(servers.keys())[0]
            first_server_config = servers[first_server_name]

            # Validate server configuration
            if not isinstance(first_server_config, dict):
                raise ValueError(
                    f'Server configuration for {first_server_name} must be a dictionary'
                )

            if 'type' not in first_server_config:
                raise ValueError(
                    f'Server type is required for {first_server_name}')

            if 'url' not in first_server_config and 'command' not in first_server_config:
                raise ValueError(
                    f'Server URL or command is required for {first_server_name}'
                )

            self.mcp_server = first_server_config
        else:
            # Direct configuration
            if 'type' not in config:
                raise ValueError('Server type is required')

            if 'url' not in config and 'command' not in config:
                raise ValueError('Server URL or command is required')

            # Validate transport type
            transport_type = config.get('type')
            if transport_type not in ['stdio', 'sse', 'streamable_http']:
                raise ValueError(
                    f'Unsupported transport type: {transport_type}')

    async def connect(self) -> None:
        """Connect to server"""
        if self.connected:
            logger.warning(f'Already connected to server {self.server_name}')
            return

        try:
            # Create new exit_stack
            self.exit_stack = AsyncExitStack()

            # Establish connection based on transport type
            if self.mcp_server['type'] == 'stdio':
                read, write = await self._establish_stdio_connection()
            elif self.mcp_server['type'] == 'sse':
                read, write = await self._establish_sse_connection()
            elif self.mcp_server['type'] == 'streamable_http':
                read, write = await self._establish_streamable_http_connection(
                )
            else:
                raise MCPConnectionError(
                    f'Unsupported transport type: {self.mcp_server["type"]}')

            # Create session
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(
                    read,
                    write,
                    client_info=self.client_info,
                    read_timeout_seconds=self.read_timeout,
                ))

            # Initialize session
            init_result = await self.session.initialize()

            # Get server information and update server name
            self._update_server_info(init_result)

            self.connected = True
            logger.info(f'Connected to server {self.server_name}')

        except Exception as e:
            logger.error(
                f'Failed to connect to server {self.server_name}: {e}')
            await self._cleanup()
            raise MCPConnectionError(f'Connection failed: {e}') from e

    async def _establish_stdio_connection(self) -> tuple[Any, Any]:
        """Establish STDIO connection"""
        config = self.mcp_server
        command = config.get('command', [])

        if not command:
            raise ValueError('STDIO command is required')

        # Create STDIO transport
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(StdioServerParameters(command=command)))
        return stdio_transport[0], stdio_transport[1]  # read, write

    async def _establish_sse_connection(self) -> tuple[Any, Any]:
        """Establish SSE connection"""
        config = self.mcp_server
        url = config.get('url')

        if not url:
            raise ValueError('SSE URL is required')

        # Create SSE transport
        sse_transport = await self.exit_stack.enter_async_context(
            sse_client(
                url,
                timeout=DEFAULT_HTTP_TIMEOUT.total_seconds(),
                sse_read_timeout=DEFAULT_SSE_READ_TIMEOUT.total_seconds()))
        return sse_transport[0], sse_transport[1]  # read, write

    async def _establish_streamable_http_connection(self) -> tuple[Any, Any]:
        """Establish Streamable HTTP connection"""
        config = self.mcp_server
        url = config.get('url')

        if not url:
            raise ValueError('Streamable HTTP URL is required')

        # Create Streamable HTTP transport
        streamable_http_transport = await self.exit_stack.enter_async_context(
            streamablehttp_client(
                url,
                timeout=DEFAULT_HTTP_TIMEOUT,
                sse_read_timeout=DEFAULT_SSE_READ_TIMEOUT))
        return streamable_http_transport[0], streamable_http_transport[
            1]  # read, write

    def _update_server_info(self, init_result) -> None:
        """Get server information from initialization result and update server name"""
        try:
            # Get server information from initialization result
            if hasattr(init_result, 'serverInfo') and init_result.serverInfo:
                self.server_info = {
                    'name': init_result.serverInfo.name,
                    'version': init_result.serverInfo.version
                }

                # If user didn't specify server name, use server's name
                if self.server_info.get('name'):
                    server_name = self.server_info['name']
                    if server_name != self.server_name:
                        logger.info(
                            f'Server name updated from "{self.server_name}" to "{server_name}"'
                        )
                        self.server_name = server_name

        except Exception as e:
            logger.warning(f'Failed to update server info: {e}')

    async def disconnect(self) -> None:
        """Disconnect from server"""
        await self._cleanup()

    async def _cleanup(self) -> None:
        """Clean up resources"""
        try:
            # Don't manually call session.close(), let AsyncExitStack handle it automatically
            if self.session:
                self.session = None

            if self.exit_stack:
                try:
                    await self.exit_stack.aclose()
                except Exception as e:
                    # Ignore cleanup errors, these are usually normal
                    logger.debug(f'Exit stack cleanup warning: {e}')
                finally:
                    self.exit_stack = None

        except Exception as e:
            logger.warning(f'Error during cleanup: {e}')
        finally:
            self.connected = False

    async def call_tool(self, tool_name: str, tool_args: Dict[str,
                                                              Any]) -> str:
        """
        Call tool on the connected MCP server

        This method executes a specific tool with given arguments and returns
        the result as a string. The tool must exist on the connected server.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Dictionary of arguments to pass to the tool

        Returns:
            Tool execution result as a string

        Raises:
            MCPConnectionError: If not connected to server or connection lost
            MCPToolExecutionError: If tool execution fails
            MCPTimeoutError: If operation times out

        Usage:

        Basic tool execution:
        >>> async with MCPClient(config) as client:
        ...     # Call tool without arguments
        ...     result = await client.call_tool("get_time", {})
        ...     print(f"Current time: {result}")
        ...     # Call tool with arguments
        ...     result = await client.call_tool("search", {
        ...             "query": "python programming",
        ...             "limit": 10
        ...         })
        ...     print(f"Search results: {result}")


        Error handling for tool calls:
        >>> async def safe_tool_call():
        ...     async with MCPClient(config) as client:
        ...         try:
        ...             result = await client.call_tool("risky_tool", {"param": "value"})
        ...             return result
        ...         except MCPToolExecutionError as e:
        ...             print(f"Tool failed: {e}")
        ...             return None
        ...         except MCPConnectionError as e:
        ...             print(f"Connection lost: {e}")
        ...             return None

        Batch tool execution:
        >>> async def call_multiple_tools():
        ...     async with MCPClient(config) as client:
        ...         tools_to_call = [
        ...             ("tool1", {"param1": "value1"}),
        ...             ("tool2", {"param2": "value2"}),
        ...             ("tool3", {"param3": "value3"})
        ...         ]
        ...
        ...         results = []
        ...         for tool_name, args in tools_to_call:
        ...             try:
        ...                 result = await client.call_tool(tool_name, args)
        ...                 results.append((tool_name, result))
        ...             except Exception as e:
        ...                 results.append((tool_name, f"Error: {e}"))
        ...
        ...         return results
        """
        if not self.connected or not self.session:
            raise MCPConnectionError(
                f'Not connected to server {self.server_name}')

        try:
            result = await self.session.call_tool(
                tool_name, tool_args, read_timeout_seconds=self.read_timeout)

            # Extract text content
            texts = []
            for content in result.content:
                if content.type == 'text':
                    texts.append(content.text)

            if texts:
                return '\n\n'.join(texts)
            else:
                return ''

        except McpError as e:
            logger.error(
                f'MCP error calling tool {tool_name} on server {self.server_name}: {e}'
            )
            if e.error.code == CONNECTION_CLOSED:
                self.connected = False
                raise MCPConnectionError(
                    f'Connection lost while calling tool {tool_name}: {e.error.message}'
                ) from e
            else:
                raise MCPToolExecutionError(
                    f'Tool execution failed: {e.error.message}') from e

        except asyncio.TimeoutError:
            raise MCPTimeoutError(
                f'Tool call {tool_name} timed out after {self.read_timeout}')

        except Exception as e:
            logger.error(
                f'Failed to call tool {tool_name} on server {self.server_name}: {e}'
            )
            raise MCPToolExecutionError(f'Tool execution failed: {e}') from e

    async def list_tools(self) -> List[Tool]:
        """
        Get list of available tools from the connected MCP server

        This method retrieves all tools that are available on the connected
        server, including their names, descriptions, and input schemas.

        Returns:
            List of Tool objects containing tool information

        Raises:
            MCPConnectionError: If not connected to server
            Exception: If operation fails

        Usage:

        Basic tool listing:
        >>> async with MCPClient(config) as client:
        ...     tools = await client.list_tools()
        ...     print(f"Found {len(tools)} tools:")
        ...     for tool in tools:
        ...         print(f"- {tool.name}: {tool.description}")


        """
        if not self.connected:
            raise MCPConnectionError('Not connected to server')

        try:
            result = await self.session.list_tools()
            return result.tools

        except Exception as e:
            logger.error(f'Failed to get tools: {e}')
            raise

    def is_connected(self) -> bool:
        """Check if connected"""
        return self.connected

    def get_server_name(self) -> str:
        """Get server name"""
        return self.server_name

    def get_transport_type(self) -> Optional[str]:
        """Get transport type"""
        return self.mcp_server.get('type')

    def get_server_info(self) -> Optional[Dict[str, Any]]:
        """Get server information"""
        return self.server_info

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()

    def __del__(self):
        """Destructor"""
        try:
            # Only clean up references, don't perform async operations
            if hasattr(self, 'session'):
                self.session = None

            if hasattr(self, 'exit_stack'):
                self.exit_stack = None

            self.connected = False

        except Exception:
            # Cannot throw exceptions in destructor
            pass
