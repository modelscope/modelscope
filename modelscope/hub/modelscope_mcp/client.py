# Copyright (c) Alibaba, Inc. and its affiliates.
"""
MCP (Model Context Protocol) API interface for ModelScope Hub.

Enhanced Single Server MCP (Model Context Protocol) Client with Multi-Transport Support.

This module provides an enhanced client implementation that connects to exactly one MCP server,
ensuring 1:1 relationship between client and server, with improved error handling,
connection management, and resource cleanup following official MCP SDK best practices.

Features:
- Multi-transport support (stdio, SSE, StreamableHTTP, WebSocket)
- Advanced component management and tool routing
- Session mapping and state tracking
- Comprehensive error handling and recovery
- Health checking and automatic reconnection
- Timeout control for all operations
- Resource management with AsyncExitStack
- LLM integration with OpenAI API
- Streaming response support
- MCP response generation with tool calling
- Enhanced concurrency support
- Improved resource cleanup
"""

import asyncio
import inspect
import logging
import os
import shutil
import time
import weakref
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.websocket import websocket_client
from mcp.shared.exceptions import McpError
from mcp.types import (CONNECTION_CLOSED, INVALID_PARAMS, INVALID_REQUEST,
                       CallToolResult, ErrorData, Implementation,
                       ListToolsResult, Tool)
from openai import OpenAI
from openai.types.chat import ChatCompletion

# Configure logging
logger = logging.getLogger(__name__)

EncodingErrorHandler = Literal['strict', 'ignore', 'replace']

DEFAULT_ENCODING = 'utf-8'
DEFAULT_ENCODING_ERROR_HANDLER: EncodingErrorHandler = 'strict'
DEFAULT_HTTP_TIMEOUT = 5
DEFAULT_SSE_READ_TIMEOOUT = 60 * 5
DEFAULT_READ_TIMEOUT = timedelta(seconds=30)
DEFAULT_CLIENT_INFO = Implementation(
    name='modelscope-mcp-single', version='0.1.0')


class MCPClientError(Exception):
    """Base exception for single server MCP client errors."""
    pass


class MCPConnectionError(MCPClientError):
    """Exception raised when connection to MCP server fails."""
    pass


class MCPToolExecutionError(MCPClientError):
    """Exception raised when tool execution fails."""
    pass


class MCPTimeoutError(MCPClientError):
    """Exception raised when operation times out."""
    pass


class MCPTransportError(MCPClientError):
    """Exception raised when transport layer fails."""
    pass


class MCPResourceError(MCPClientError):
    """Exception raised when resource management fails."""
    pass


class TransportType:
    """Transport type constants."""
    STDIO = 'stdio'
    SSE = 'sse'
    STREAMABLE_HTTP = 'streamablehttp'
    WEBSOCKET = 'websocket'


class ComponentManager:
    """
    Advanced component management for MCP client.

    Manages tools, resources, and prompts with proper mapping and routing.
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.resources: Dict[str, Any] = {}
        self.prompts: Dict[str, Any] = {}
        self.tool_to_session: Dict[str, str] = {}  # tool_name -> session_id
        self.session_components: Dict[str, Dict[str, List[str]]] = {
        }  # session_id -> {component_type: [names]}

    def add_tool(self, tool: Tool, session_id: str) -> None:
        """Add a tool to the component manager."""
        self.tools[tool.name] = tool
        self.tool_to_session[tool.name] = session_id

        if session_id not in self.session_components:
            self.session_components[session_id] = {
                'tools': [],
                'resources': [],
                'prompts': []
            }
        self.session_components[session_id]['tools'].append(tool.name)

    def get_tool_session(self, tool_name: str) -> Optional[str]:
        """Get the session ID for a specific tool."""
        return self.tool_to_session.get(tool_name)

    def get_session_tools(self, session_id: str) -> List[Tool]:
        """Get all tools for a specific session."""
        if session_id not in self.session_components:
            return []
        tool_names = self.session_components[session_id].get('tools', [])
        return [self.tools[name] for name in tool_names if name in self.tools]

    def clear_session_components(self, session_id: str) -> None:
        """Clear all components for a specific session."""
        if session_id in self.session_components:
            # Remove tools
            for tool_name in self.session_components[session_id].get(
                    'tools', []):
                self.tools.pop(tool_name, None)
                self.tool_to_session.pop(tool_name, None)
            # Clear session components
            self.session_components.pop(session_id, None)

    def get_all_tools(self) -> List[Tool]:
        """Get all tools from component manager."""
        return list(self.tools.values())

    def get_tool_count(self) -> int:
        """Get the number of tools."""
        return len(self.tools)


class MCPClient:
    """
    Enhanced Single Server MCP (Model Context Protocol) Client with Multi-Transport Support.

    This client connects to exactly one MCP server, ensuring 1:1 relationship
    between client and server for better resource isolation and clear communication.

    Features:
    - Multi-transport support (stdio, SSE, StreamableHTTP, WebSocket)
    - Advanced component management and tool routing
    - Session mapping and state tracking
    - Robust error handling following official MCP SDK patterns
    - Connection lifecycle management with proper cleanup
    - Health checking and automatic reconnection
    - Timeout control for all operations
    - Resource management with AsyncExitStack
    - LLM integration with OpenAI API
    - Streaming response support
    - MCP response generation with tool calling
    - Enhanced concurrency support
    - Improved resource cleanup
    """

    def __init__(
        self,
        server_name: str,
        server_config: Dict[str, Any],
        client_info: Optional[Implementation] = None,
        read_timeout: Optional[timedelta] = None,
        health_check_interval: int = 30,
        max_reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
        component_name_hook: Optional[Callable[[str, Implementation],
                                               str]] = None,
    ) -> None:
        """
        Initialize MCP Client with Multi-Transport Support.

        Args:
            server_name: Name of the server this client connects to
            server_config: Configuration for the single server
            client_info: Optional client implementation info
            read_timeout: Timeout for read operations
            health_check_interval: Interval for health checks in seconds
            max_reconnect_attempts: Maximum number of reconnection attempts
            reconnect_delay: Delay between reconnection attempts in seconds
            component_name_hook: Optional hook for custom component naming

        Raises:
            ValueError: If required configuration is missing
        """
        if not server_name:
            raise ValueError('Server name is required')
        if not server_config:
            raise ValueError('Server configuration is required')

        self.server_name = server_name
        self.server_config = server_config
        self.session: Optional[ClientSession] = None
        self.session_id: Optional[str] = None
        self.exit_stack = AsyncExitStack()
        self.client_info = client_info or DEFAULT_CLIENT_INFO
        self.connected = False
        self.read_timeout = read_timeout or DEFAULT_READ_TIMEOUT
        self.health_check_interval = health_check_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.component_name_hook = component_name_hook

        # Advanced component management
        self.component_manager = ComponentManager()

        # Connection state tracking
        self._connection_attempts = 0
        self._last_health_check = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._transport_type: Optional[str] = None

        # Silent initialization, no logging

    @staticmethod
    def validate_server_config(
            server_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate server configuration with multi-transport support.

        Args:
            server_config: Raw server configuration

        Returns:
            Validated configuration

        Raises:
            FileNotFoundError: If required commands are not found
            ValueError: If configuration is invalid
        """
        if not server_config:
            raise ValueError('Server configuration cannot be empty')

        if not isinstance(server_config, dict):
            raise ValueError('Server configuration must be a dictionary')

        # 检查服务类型
        if 'command' in server_config:
            # 本地进程服务 (stdio 连接)
            command = server_config['command']

            # Handle special commands following official patterns
            if 'fastmcp' in command:
                command = shutil.which('fastmcp')
                if not command:
                    raise FileNotFoundError(
                        'Cannot locate the fastmcp command file, please install fastmcp by `pip install fastmcp`'
                    )
                server_config['command'] = command

            if 'uv' in command:
                command = shutil.which('uv')
                if not command:
                    raise FileNotFoundError(
                        'Cannot locate the uv command, please consider your installation of Python.'
                    )

            # Process arguments
            args = server_config.get('args', [])
            if not isinstance(args, list):
                raise ValueError("Invalid 'args' in server configuration")

            for idx, arg in enumerate(args):
                if isinstance(arg, str) and '/path/to' in arg:
                    # TODO: Further integration needed for stdio tools
                    args[idx] = arg.replace('/path/to', os.getcwd())

        elif 'url' in server_config:
            # Remote service (SSE/HTTP/WebSocket connection)
            url = server_config['url']
            if not isinstance(url, str):
                raise ValueError("Invalid 'url' in server configuration")

            # Check service type
            service_type = server_config.get('type', 'http')
            if service_type not in [
                    'sse', 'http', 'streamablehttp', 'websocket'
            ]:
                raise ValueError(f"Unsupported service type '{service_type}'")

        else:
            # Invalid configuration
            raise ValueError(
                "Server must have either 'command' (for local process) "
                "or 'url' (for remote service) field")

        return server_config

    @staticmethod
    def parse_config(mcp_servers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and validate MCP server configuration.

        Args:
            mcp_servers: Raw MCP server configuration

        Returns:
            Parsed and validated configuration

        Raises:
            FileNotFoundError: If required commands are not found
            ValueError: If configuration is invalid
        """
        if not mcp_servers:
            raise ValueError('MCP servers configuration cannot be empty')

        config_json = {}
        for mcp_server_name, mcp_content in mcp_servers.items():
            if not isinstance(mcp_content, dict):
                raise ValueError(
                    f'Invalid configuration for server {mcp_server_name}')

            # Check service type
            if 'command' in mcp_content:
                # Local process service (stdio connection)
                command = mcp_content['command']

                # Handle special commands
                if 'fastmcp' in command:
                    command = shutil.which('fastmcp')
                    if not command:
                        raise FileNotFoundError(
                            'Cannot locate the fastmcp command file, please install fastmcp by `pip install fastmcp`'
                        )
                    mcp_content['command'] = command

                if 'uv' in command:
                    command = shutil.which('uv')
                    if not command:
                        raise FileNotFoundError(
                            'Cannot locate the uv command, please consider your installation of Python.'
                        )

                # Process arguments
                args = mcp_content.get('args', [])
                if not isinstance(args, list):
                    raise ValueError(
                        f"Invalid 'args' in configuration for server {mcp_server_name}"
                    )

                for idx, arg in enumerate(args):
                    if isinstance(arg, str) and '/path/to' in arg:
                        # TODO: Further integration needed for stdio tools
                        args[idx] = arg.replace('/path/to', os.getcwd())

            elif 'url' in mcp_content:
                # Remote service (SSE/HTTP connection)
                url = mcp_content['url']
                if not isinstance(url, str):
                    raise ValueError(
                        f"Invalid 'url' in configuration for server {mcp_server_name}"
                    )

                # Check service type
                service_type = mcp_content.get('type', 'http')
                if service_type not in [
                        'sse', 'http', 'streamablehttp', 'websocket'
                ]:
                    raise ValueError(
                        f"Unsupported service type '{service_type}' for server {mcp_server_name}"
                    )

            else:
                # Invalid configuration
                raise ValueError(
                    f"Server {mcp_server_name} must have either 'command' (for local process) "
                    f"or 'url' (for remote service) field")

            config_json[mcp_server_name] = mcp_content

        return config_json

    @staticmethod
    def generate_config(mcp_servers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate MCP configuration from server definitions.

        Args:
            mcp_servers: Server definitions

        Returns:
            Generated configuration
        """
        config = {}
        for server_name, server_config in mcp_servers.items():
            if isinstance(server_config, dict) and 'command' in server_config:
                config[server_name] = server_config.copy()
            else:
                # Handle simple string configurations
                config[server_name] = {
                    'command': server_config,
                    'args': [],
                    'env': {}
                }
        return config

    def _determine_transport_type(self) -> str:
        """Determine the transport type from server configuration."""
        if 'command' in self.server_config:
            return TransportType.STDIO
        elif 'url' in self.server_config:
            service_type = self.server_config.get('type', 'http')
            if service_type == 'websocket':
                return TransportType.WEBSOCKET
            elif service_type == 'streamablehttp':
                return TransportType.STREAMABLE_HTTP
            else:
                return TransportType.SSE
        else:
            raise ValueError(
                'Cannot determine transport type from configuration')

    async def _establish_stdio_connection(self) -> tuple[Any, Any]:
        """Establish stdio connection."""
        server_params = StdioServerParameters(
            command=self.server_config['command'],
            args=self.server_config.get('args', []),
            env={
                **os.environ,
                **self.server_config.get('env', {})
            },
            encoding=self.server_config.get('encoding', DEFAULT_ENCODING),
            encoding_error_handler=self.server_config.get(
                'encoding_error_handler', DEFAULT_ENCODING_ERROR_HANDLER))

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params))
        return stdio_transport

    async def _establish_sse_connection(self) -> tuple[Any, Any]:
        """Establish SSE connection."""
        url = self.server_config['url']
        sse_transport = await self.exit_stack.enter_async_context(
            sse_client(
                url,
                timeout=self.server_config.get('timeout',
                                               DEFAULT_HTTP_TIMEOUT),
                sse_read_timeout=self.server_config.get(
                    'sse_read_timeout', DEFAULT_SSE_READ_TIMEOOUT)))
        return sse_transport

    async def _establish_streamablehttp_connection(
            self) -> tuple[Any, Any, Callable]:
        """Establish StreamableHTTP connection."""
        url = self.server_config['url']
        streamablehttp_transport = await self.exit_stack.enter_async_context(
            streamablehttp_client(
                url,
                timeout=self.server_config.get('timeout',
                                               DEFAULT_HTTP_TIMEOUT),
                sse_read_timeout=self.server_config.get(
                    'sse_read_timeout', DEFAULT_SSE_READ_TIMEOOUT)))
        return streamablehttp_transport

    async def _establish_websocket_connection(self) -> tuple[Any, Any]:
        """Establish WebSocket connection."""
        url = self.server_config['url']
        websocket_transport = await self.exit_stack.enter_async_context(
            websocket_client(url))
        return websocket_transport

    async def _run_in_isolated_context(self, coro):
        """Run coroutine in isolated task context, ensuring resource creation and cleanup in the same context"""
        try:
            # Create independent task
            task = asyncio.create_task(coro)
            return await task
        except Exception as e:
            logger.error(f'Error in isolated context: {e}')
            raise

    async def _connect_in_isolated_context(self):
        """Execute connection in isolated context"""
        if self.connected:
            logger.warning(f'Already connected to server {self.server_name}')
            return

        # Ensure each connection uses a new exit_stack
        if hasattr(self, 'exit_stack') and self.exit_stack:
            try:
                await self._safe_close_exit_stack(self.exit_stack)
            except Exception:
                pass  # Ignore already closed exit_stack errors

        # Create new exit_stack
        self.exit_stack = AsyncExitStack()
        # Ensure old references are cleaned up
        import gc
        gc.collect()

        # Validate configuration
        self.server_config = self.validate_server_config(self.server_config)
        self._transport_type = self._determine_transport_type()

        for attempt in range(self.max_reconnect_attempts):
            try:
                self._connection_attempts += 1

                # Establish transport connection based on type
                if self._transport_type == TransportType.STDIO:
                    read, write = await self._establish_stdio_connection()
                elif self._transport_type == TransportType.SSE:
                    read, write = await self._establish_sse_connection()
                elif self._transport_type == TransportType.STREAMABLE_HTTP:
                    read, write, get_session_id = await self._establish_streamablehttp_connection(
                    )
                    self.session_id = get_session_id()
                elif self._transport_type == TransportType.WEBSOCKET:
                    read, write = await self._establish_websocket_connection()
                else:
                    raise MCPTransportError(
                        f'Unsupported transport type: {self._transport_type}')

                # Create session with timeout
                self.session = await self.exit_stack.enter_async_context(
                    ClientSession(
                        read,
                        write,
                        client_info=self.client_info,
                        read_timeout_seconds=self.read_timeout,
                    ))

                # Initialize session
                await self.session.initialize()

                # Generate session ID if not provided
                if not self.session_id:
                    self.session_id = f'{self.server_name}-{int(time.time())}'

                # Initialize component manager for this session
                self.component_manager.clear_session_components(
                    self.session_id)

                self.connected = True
                self._connection_attempts = 0
                self._last_health_check = time.time()

                # Start health check if enabled
                if self.health_check_interval > 0:
                    self._start_health_check()

                return

            except McpError as e:
                logger.error(
                    f'MCP protocol error connecting to {self.server_name}: {e}'
                )
                if e.error.code == CONNECTION_CLOSED:
                    raise MCPConnectionError(
                        f'Connection to {self.server_name} was closed: {e.error.message}'
                    ) from e
                else:
                    raise MCPConnectionError(
                        f'MCP error connecting to {self.server_name}: {e.error.message}'
                    ) from e

            except Exception as e:
                logger.error(
                    f'Failed to connect to server {self.server_name} (attempt {attempt + 1}): {e}'
                )
                if attempt < self.max_reconnect_attempts - 1:
                    await asyncio.sleep(self.reconnect_delay * (attempt + 1))
                else:
                    self.connected = False
                    raise MCPConnectionError(
                        f'Connection to {self.server_name} failed after {self.max_reconnect_attempts} attempts: {e}'
                    ) from e

    async def _disconnect_in_isolated_context(self):
        """Execute disconnection in isolated context"""
        if not self.connected:
            return

        # Stop health check
        self._stop_health_check()

        try:
            # Clear session components
            if self.session_id:
                self.component_manager.clear_session_components(
                    self.session_id)

            # Use safe exit_stack cleanup method
            if hasattr(self, 'exit_stack') and self.exit_stack:
                await self._safe_close_exit_stack(self.exit_stack)

            self.session = None
            self.session_id = None
            self.connected = False
        except Exception as e:
            logger.warning(f'Error during disconnect: {e}')
            # Ensure state reset
            self.session = None
            self.session_id = None
            self.connected = False

    async def connect(self) -> None:
        """
        Connect to the single MCP server with multi-transport support.

        Raises:
            MCPConnectionError: If connection fails
            MCPTransportError: If transport setup fails
            ValueError: If server configuration is invalid
        """
        await self._run_in_isolated_context(
            self._connect_in_isolated_context())

    async def disconnect(self) -> None:
        """Disconnect from the server with proper cleanup."""
        await self._run_in_isolated_context(
            self._disconnect_in_isolated_context())

    async def reconnect(self) -> None:
        """Reconnect to the server with proper error handling."""
        logger.info(f'Reconnecting to server {self.server_name}...')
        await self.disconnect()
        await self.connect()

    async def call_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        timeout: Optional[timedelta] = None,
    ) -> str:
        """
        Call a specific tool on the connected server with advanced routing.

        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool
            timeout: Optional timeout for the tool call

        Returns:
            Tool execution result

        Raises:
            MCPConnectionError: If not connected
            MCPToolExecutionError: If tool execution fails
            MCPTimeoutError: If operation times out
        """
        if not self.connected or not self.session:
            raise MCPConnectionError(
                f'Not connected to server {self.server_name}')

        try:
            # Use provided timeout or default
            read_timeout = timeout or self.read_timeout

            result = await self.session.call_tool(
                tool_name, tool_args, read_timeout_seconds=read_timeout)

            # Extract text content from result
            texts = []
            for content in result.content:
                if content.type == 'text':
                    texts.append(content.text)

            if texts:
                return '\n\n'.join(texts)
            else:
                return 'execute error'

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
                f'Tool call {tool_name} timed out after {timeout or self.read_timeout}'
            )

        except Exception as e:
            logger.error(
                f'Failed to call tool {tool_name} on server {self.server_name}: {e}'
            )
            raise MCPToolExecutionError(f'Tool execution failed: {e}') from e

    async def get_tools(self,
                        timeout: Optional[timedelta] = None) -> List[Tool]:
        """
        Get all available tools from the server.

        Args:
            timeout: Optional timeout for the operation

        Returns:
            List of available tools

        Raises:
            MCPConnectionError: If not connected
            MCPTimeoutError: If operation times out
        """
        if not self.connected:
            raise MCPConnectionError('Not connected to server')

        try:
            result = await self.session.list_tools()

            # Update component manager
            for tool in result.tools:
                if self.session_id:  # Ensure session_id is not None
                    self.component_manager.add_tool(tool, self.session_id)

            return result.tools

        except Exception as e:
            logger.error(f'Failed to get tools: {e}')
            raise

    def get_available_tools(self) -> List[Tool]:
        """Get all available tools from component manager."""
        return self.component_manager.get_all_tools()

    def get_tool_info(self, tool_name: str) -> Optional[Tool]:
        """Get information about a specific tool."""
        return self.component_manager.tools.get(tool_name)

    def is_connected(self) -> bool:
        """Check if connected to the server."""
        return self.connected

    def get_server_name(self) -> str:
        """Get the name of the server this client connects to."""
        return self.server_name

    def get_transport_type(self) -> Optional[str]:
        """Get the current transport type."""
        return self._transport_type

    def get_session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self.session_id

    def get_server_info(self) -> Dict[str, Any]:
        """Get comprehensive server information."""
        return {
            'name': self.server_name,
            'connected': self.connected,
            'transport_type': self._transport_type,
            'session_id': self.session_id,
            'config': self.server_config,
            'connection_attempts': self._connection_attempts,
            'last_health_check': self._last_health_check,
            'health_check_interval': self.health_check_interval,
            'read_timeout': str(self.read_timeout),
            'available_tools': self.component_manager.get_tool_count(),
            'component_manager_stats': {
                'tools': self.component_manager.get_tool_count(),
                'resources': len(self.component_manager.resources),
                'prompts': len(self.component_manager.prompts),
                'sessions': len(self.component_manager.session_components),
            }
        }

    async def health_check(self) -> bool:
        """
        Perform health check on the server.

        Returns:
            True if server is healthy, False otherwise
        """
        if not self.connected:
            return False

        try:
            # Simple health check: try to get tools
            await self.get_tools()
            self._last_health_check = time.time()
            return True
        except Exception as e:
            logger.warning(
                f'Health check failed for server {self.server_name}: {e}')
            return False

    def _start_health_check(self) -> None:
        """Start periodic health checking."""
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(
                self._health_check_loop())
            logger.debug(
                f'Started health check loop for server {self.server_name}')

    def _stop_health_check(self) -> None:
        """Stop periodic health checking."""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            logger.debug(
                f'Stopped health check loop for server {self.server_name}')

    async def _health_check_loop(self) -> None:
        """Health check loop that runs periodically."""
        while not self._shutdown_event.is_set() and self.connected:
            try:
                await asyncio.sleep(self.health_check_interval)

                if not self.connected:
                    break

                is_healthy = await self.health_check()
                if not is_healthy:
                    logger.warning(
                        f'Health check failed for server {self.server_name}, attempting reconnection'
                    )
                    try:
                        await self.reconnect()
                    except Exception as e:
                        logger.error(
                            f'Failed to reconnect to server {self.server_name}: {e}'
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f'Error in health check loop for server {self.server_name}: {e}'
                )

    async def ping(self) -> bool:
        """
        Send a ping to the server to check connectivity.

        Returns:
            True if ping successful, False otherwise
        """
        if not self.connected or not self.session:
            return False

        try:
            await self.session.send_ping()
            return True
        except Exception as e:
            logger.warning(f'Ping failed for server {self.server_name}: {e}')
            return False

    async def _cleanup_in_isolated_context(self):
        """Execute cleanup in isolated context"""
        try:
            await self._disconnect_in_isolated_context()

            # Additional resource cleanup
            if hasattr(self, 'exit_stack') and self.exit_stack:
                await self._safe_close_exit_stack(self.exit_stack)
                self.exit_stack = None

            # Clean up component manager
            if hasattr(self, 'component_manager'):
                self.component_manager = None

            # Force garbage collection
            import gc
            gc.collect()

            # Silent cleanup completion, no logging
        except asyncio.CancelledError:
            # Handle cancellation exception
            logger.info(f'Cleanup cancelled for {self.server_name}')
            # Ensure state reset
            self.session = None
            self.session_id = None
            self.connected = False
            self.exit_stack = None
            # Don't re-raise exception
        except Exception as e:
            logger.warning(f'Error during cleanup: {e}')
            # Ensure state reset
            self.session = None
            self.session_id = None
            self.connected = False
            self.exit_stack = None
            # Don't re-raise exception

    async def cleanup(self) -> None:
        """Clean up all connections and resources."""
        await self._run_in_isolated_context(
            self._cleanup_in_isolated_context())

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper cleanup."""
        await self.disconnect()

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            # Cancel health check task
            if hasattr(
                    self, '_health_check_task'
            ) and self._health_check_task and not self._health_check_task.done(
            ):
                self._health_check_task.cancel()

            # Clean up exit_stack references
            if hasattr(self, 'exit_stack') and self.exit_stack:
                # Cannot use await in destructor, only clean up references
                self.exit_stack._stack = []
                self.exit_stack._closed = True
                self.exit_stack = None

            # Clean up other references
            self.session = None
            self.session_id = None
            self.connected = False

        except Exception:
            # Cannot raise exceptions in destructor
            pass

    async def _force_close_exit_stack(self, exit_stack):
        """Force cleanup AsyncExitStack, ensuring all resources are released"""
        if not exit_stack:
            return
        try:
            # Use safer way to access internal state
            if hasattr(exit_stack, '_stack'):
                # Manually trigger __aexit__ or __exit__ for all resources
                for item in list(
                        exit_stack._stack
                ):  # Copy list to avoid modification during iteration
                    try:
                        if hasattr(item, '__aexit__'):
                            await item.__aexit__(None, None, None)
                        elif hasattr(item, '__exit__'):
                            item.__exit__(None, None, None)
                    except Exception:
                        # Silently handle resource close errors
                        pass
            else:
                # If no _stack attribute, try other methods
                # Try direct aclose call, even if it might fail
                try:
                    await exit_stack.aclose()
                except Exception:
                    # Silently handle fallback cleanup errors
                    pass
        except Exception:
            # Silently handle force cleanup errors
            pass
        finally:
            # Try to reset exit_stack state
            try:
                if hasattr(exit_stack, '_stack'):
                    exit_stack._stack = []
                if hasattr(exit_stack, '_closed'):
                    exit_stack._closed = True
            except Exception:
                # Silently handle state reset errors
                pass

    async def _safe_close_exit_stack(self, exit_stack):
        """Safely close exit_stack, combining multiple cleanup methods"""
        if not exit_stack:
            return
        try:
            # First try normal aclose
            await exit_stack.aclose()
        except Exception:
            # Silently handle cleanup errors, no warnings
            pass
        finally:
            # Ensure references are cleaned up
            try:
                if hasattr(exit_stack, '_stack'):
                    exit_stack._stack = []
                if hasattr(exit_stack, '_closed'):
                    exit_stack._closed = True
            except Exception:
                # Silently handle final cleanup errors
                pass
