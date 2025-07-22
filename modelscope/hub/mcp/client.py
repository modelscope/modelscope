#!/usr/bin/env python3
"""
MCP客户端 - 基于官方 MCP Python SDK 的简洁实现
"""

import asyncio
import time
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any, Dict, List, Optional

# 导入官方 MCP SDK
from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED, Implementation

from modelscope.utils.logger import get_logger

# 常量定义
DEFAULT_CLIENT_INFO = Implementation(
    name='modelscope-mcp-client', version='1.0.0')

DEFAULT_READ_TIMEOUT = timedelta(seconds=30)
DEFAULT_HTTP_TIMEOUT = timedelta(seconds=30)
DEFAULT_SSE_READ_TIMEOUT = timedelta(seconds=30)

# 日志
logger = get_logger(__name__)


# 异常类
class MCPClientError(Exception):
    """MCP客户端基础异常"""
    pass


class MCPConnectionError(MCPClientError):
    """MCP连接异常"""
    pass


class MCPToolExecutionError(MCPClientError):
    """MCP工具执行异常"""
    pass


class MCPTimeoutError(MCPClientError):
    """MCP超时异常"""
    pass


class MCPClient:
    """
    MCP客户端 - 基于官方 MCP Python SDK 的简洁实现

    最简单的使用方法:
    ```python
    # 1. 创建客户端
    client = MCPClient(mcp_server={
        "type": "sse",
        "url": "https://example.com/sse"
    })

    # 2. 连接并使用
    await client.connect()
    tools = await client.list_tools()
    result = await client.call_tool("tool_name", {"param": "value"})
    await client.disconnect()

    # 或者使用上下文管理器（推荐）
    async with MCPClient(mcp_server=config) as client:
        tools = await client.list_tools()
        result = await client.call_tool("tool_name", {"param": "value"})
    ```
    """

    def __init__(self, mcp_server: Dict[str, Any]):
        """
        初始化 MCP 客户端

        Args:
            mcp_server: MCP 服务器配置
        """
        if not mcp_server:
            raise ValueError('MCP server configuration is required')

        self.mcp_server = mcp_server
        self.session: Optional[ClientSession] = None
        self.exit_stack: Optional[AsyncExitStack] = None
        self.client_info = DEFAULT_CLIENT_INFO
        self.connected = False
        self.read_timeout = DEFAULT_READ_TIMEOUT
        self.server_info: Optional[Dict[str, Any]] = None  # 服务器信息

        # 验证配置
        self._validate_config()

        # 自动生成服务器名称（连接后可能会更新）
        self.server_name = self._generate_server_name()

    def _generate_server_name(self) -> str:
        """自动生成服务器名称"""
        config = self.mcp_server

        # 从配置中提取有意义的名称
        if 'type' in config:
            transport_type = config['type']

            if transport_type == 'stdio' and 'command' in config:
                # 从命令中提取名称
                command = config['command']
                if isinstance(command, list) and command:
                    return f'stdio-{command[0]}'
                elif isinstance(command, str):
                    return f'stdio-{command}'

            elif transport_type in ['sse', 'streamable_http'
                                    ] and 'url' in config:
                # 从 URL 中提取域名
                url = config['url']
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    domain = parsed.netloc.split('.')[0]  # 取第一个域名部分
                    return f'{transport_type}-{domain}'
                except Exception:
                    return f'{transport_type}-server'

                # 默认名称
        return f"mcp-{config.get('type', 'unknown')}-server"

    def _validate_config(self) -> None:
        """验证 MCP 服务器配置"""
        config = self.mcp_server

        # 检查是否有 mcpServers 嵌套结构
        if 'mcpServers' in config:
            servers = config['mcpServers']
            if not servers:
                raise ValueError('No servers found in mcpServers')

            # 获取第一个服务器的配置
            first_server_name = list(servers.keys())[0]
            first_server_config = servers[first_server_name]

            # 验证服务器配置
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
            # 直接配置
            if 'type' not in config:
                raise ValueError('Server type is required')

            if 'url' not in config and 'command' not in config:
                raise ValueError('Server URL or command is required')

            # 验证传输类型
            transport_type = config.get('type')
            if transport_type not in ['stdio', 'sse', 'streamable_http']:
                raise ValueError(
                    f'Unsupported transport type: {transport_type}')

    async def connect(self) -> None:
        """连接到服务器"""
        if self.connected:
            logger.warning(f'Already connected to server {self.server_name}')
            return

        try:
            # 创建新的 exit_stack
            self.exit_stack = AsyncExitStack()

            # 根据传输类型建立连接
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

            # 创建会话
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(
                    read,
                    write,
                    client_info=self.client_info,
                    read_timeout_seconds=self.read_timeout,
                ))

            # 初始化会话
            init_result = await self.session.initialize()

            # 获取服务器信息并更新服务器名称
            self._update_server_info(init_result)

            self.connected = True
            logger.info(f'Connected to server {self.server_name}')

        except Exception as e:
            logger.error(
                f'Failed to connect to server {self.server_name}: {e}')
            await self._cleanup()
            raise MCPConnectionError(f'Connection failed: {e}') from e

    async def _establish_stdio_connection(self) -> tuple[Any, Any]:
        """建立 STDIO 连接"""
        config = self.mcp_server
        command = config.get('command', [])

        if not command:
            raise ValueError('STDIO command is required')

        # 创建 STDIO 传输
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(StdioServerParameters(command=command)))
        return stdio_transport[0], stdio_transport[1]  # read, write

    async def _establish_sse_connection(self) -> tuple[Any, Any]:
        """建立 SSE 连接"""
        config = self.mcp_server
        url = config.get('url')

        if not url:
            raise ValueError('SSE URL is required')

        # 创建 SSE 传输
        sse_transport = await self.exit_stack.enter_async_context(
            sse_client(
                url,
                timeout=DEFAULT_HTTP_TIMEOUT.total_seconds(),
                sse_read_timeout=DEFAULT_SSE_READ_TIMEOUT.total_seconds()))
        return sse_transport[0], sse_transport[1]  # read, write

    async def _establish_streamable_http_connection(self) -> tuple[Any, Any]:
        """建立 Streamable HTTP 连接"""
        config = self.mcp_server
        url = config.get('url')

        if not url:
            raise ValueError('Streamable HTTP URL is required')

        # 创建 Streamable HTTP 传输
        streamable_http_transport = await self.exit_stack.enter_async_context(
            streamablehttp_client(
                url,
                timeout=DEFAULT_HTTP_TIMEOUT,
                sse_read_timeout=DEFAULT_SSE_READ_TIMEOUT))
        return streamable_http_transport[0], streamable_http_transport[
            1]  # read, write

    def _update_server_info(self, init_result) -> None:
        """从初始化结果中获取服务器信息并更新服务器名称"""
        try:
            # 从初始化结果中获取服务器信息
            if hasattr(init_result, 'serverInfo') and init_result.serverInfo:
                self.server_info = {
                    'name': init_result.serverInfo.name,
                    'version': init_result.serverInfo.version
                }

                # 如果用户没有指定服务器名称，使用服务器的名称
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
        """断开连接"""
        await self._cleanup()

    async def _cleanup(self) -> None:
        """清理资源"""
        try:
            # 不要手动调用 session.close()，让 AsyncExitStack 自动处理
            if self.session:
                self.session = None

            if self.exit_stack:
                try:
                    await self.exit_stack.aclose()
                except Exception as e:
                    # 忽略清理时的错误，这些通常是正常的
                    logger.debug(f'Exit stack cleanup warning: {e}')
                finally:
                    self.exit_stack = None

        except Exception as e:
            logger.warning(f'Error during cleanup: {e}')
        finally:
            self.connected = False

    async def call_tool(self,
                        tool_name: str,
                        tool_args: Dict[str, Any],
                        timeout: Optional[timedelta] = None) -> str:
        """
        调用工具

        Args:
            tool_name: 工具名称
            tool_args: 工具参数
            timeout: 超时时间

        Returns:
            工具执行结果
        """
        if not self.connected or not self.session:
            raise MCPConnectionError(
                f'Not connected to server {self.server_name}')

        try:
            read_timeout = timeout or self.read_timeout

            result = await self.session.call_tool(
                tool_name, tool_args, read_timeout_seconds=read_timeout)

            # 提取文本内容
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

    async def list_tools(self,
                         timeout: Optional[timedelta] = None) -> List[Tool]:
        """
        获取工具列表

        Args:
            timeout: 超时时间

        Returns:
            工具列表
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
        """检查是否已连接"""
        return self.connected

    def get_server_name(self) -> str:
        """获取服务器名称"""
        return self.server_name

    def get_transport_type(self) -> Optional[str]:
        """获取传输类型"""
        return self.mcp_server.get('type')

    def get_server_info(self) -> Optional[Dict[str, Any]]:
        """获取服务器信息"""
        return self.server_info

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.disconnect()

    def __del__(self):
        """析构函数"""
        try:
            # 只清理引用，不进行异步操作
            if hasattr(self, 'session'):
                self.session = None

            if hasattr(self, 'exit_stack'):
                self.exit_stack = None

            self.connected = False

        except Exception:
            # 析构函数中不能抛出异常
            pass
