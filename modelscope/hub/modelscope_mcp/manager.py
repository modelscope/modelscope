# Copyright (c) Alibaba, Inc. and its affiliates.
"""
MCP (Model Context Protocol) API interface for ModelScope Hub.

New MCP (Model Context Protocol) Manager for ModelScope.

This module provides a manager that coordinates multiple SingleServerMCPClient instances,
offering a unified interface for tool management and execution.

Design Philosophy:
- Each SingleServerMCPClient connects to exactly one MCP server (1:1 relationship)
- MCPManager coordinates multiple clients and provides unified tool interface
- Clean separation of concerns: client handles connection, manager handles coordination
- Thread-safe operations with proper resource management
- Integrated with ModelScope authentication system
"""

import asyncio
# Add log level control at file beginning
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Union

import json

from modelscope.hub.api import HubApi, ModelScopeConfig
from modelscope.hub.constants import (DEFAULT_MODELSCOPE_DATA_ENDPOINT,
                                      DEFAULT_MODELSCOPE_INTL_DATA_ENDPOINT)
from modelscope.hub.modelscope_mcp.api import McpApi
from modelscope.hub.modelscope_mcp.client import (MCPClient, MCPClientError,
                                                  MCPConnectionError,
                                                  MCPToolExecutionError)
from modelscope.hub.modelscope_mcp.config_manager import (load_config,
                                                          merge_configs)
from modelscope.hub.modelscope_mcp.types import (APIConfig, MCPConfig,
                                                 create_client_info,
                                                 validate_api_config)
from modelscope.hub.modelscope_mcp.utils import (MCPConfigError,
                                                 format_mcp_tool_name,
                                                 parse_mcp_tool_name,
                                                 validate_mcp_config)
from modelscope.hub.utils.utils import get_endpoint

# Set log level to WARNING to reduce detailed log output
logging.getLogger('modelscope.hub.modelscope_mcp.manager').setLevel(
    logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

# Configure logging
logger = logging.getLogger(__name__)

# Default tool exclusions
DEFAULT_TOOL_EXCLUDES: List[Union[str, Dict[str, List[str]]]] = [{
    'amap-maps': ['maps_geo']
}]

# ModelScope API configuration
MODELSCOPE_API_BASE_URL = 'https://api-inference.modelscope.cn/v1/'
MODELSCOPE_INTL_API_BASE_URL = 'https://api-inference.modelscope.ai/v1/'


class MCPManagerError(Exception):
    """Base exception for MCP Manager errors."""
    pass


class MCPManagerInitializationError(MCPManagerError):
    """Exception raised when MCP Manager initialization fails."""
    pass


class MCPManagerToolExecutionError(MCPManagerError):
    """Exception raised when MCP tool execution fails."""
    pass


class MCPManagerConnectionError(MCPManagerError):
    """Exception raised when MCP connection fails."""
    pass


@dataclass
class ToolInfo:
    """Information about an MCP tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    server_name: str
    tool_name: str
    is_available: bool = True
    last_used: Optional[float] = None
    usage_count: int = 0


@dataclass
class ServerInfo:
    """Information about an MCP server."""
    name: str
    config: Dict[str, Any]
    tools: List[ToolInfo] = field(default_factory=list)
    error_count: int = 0


class MCPTool:
    """
    Unified MCP tool interface.

    This class provides a unified interface for all MCP tools,
    regardless of which server they belong to.
    """

    def __init__(self, tool_info: ToolInfo, manager: 'MCPManager') -> None:
        """
        Initialize MCP tool.

        Args:
            tool_info: Tool information
            manager: Reference to the manager
        """
        self.tool_info = tool_info
        self._manager = manager
        self.name = tool_info.name
        self.description = tool_info.description
        self.parameters = tool_info.parameters

    def call(self, params: Union[str, Dict[str, Any]], **kwargs: Any) -> str:
        """
        Execute the tool with given parameters.

        Args:
            params: Tool parameters as string or dictionary
            **kwargs: Additional keyword arguments

        Returns:
            Tool execution result as string

        Raises:
            MCPManagerToolExecutionError: If tool execution fails
        """
        try:
            # Parse parameters if they're provided as a string
            if isinstance(params, str):
                tool_args = json.loads(params)
            else:
                tool_args = params

            # Validate parameters schema
            if not isinstance(tool_args, dict):
                raise MCPManagerToolExecutionError(
                    'Tool arguments must be a dictionary')

            # Update tool usage statistics
            self.tool_info.last_used = time.time()
            self.tool_info.usage_count += 1

            # Check if loop is available
            if self._manager.loop is None:
                raise MCPManagerToolExecutionError(
                    'Event loop is not available')

            # Submit coroutine to the event loop and wait for the result
            future = asyncio.run_coroutine_threadsafe(
                self._manager._call_tool_async(self.tool_info.server_name,
                                               self.tool_info.tool_name,
                                               tool_args), self._manager.loop)

            result = future.result()
            return result

        except json.JSONDecodeError as e:
            logger.error(f'Failed to parse tool parameters: {e}')
            raise MCPManagerToolExecutionError(
                f'Invalid JSON parameters: {e}') from e
        except Exception as e:
            logger.error(f'Failed to execute MCP tool {self.name}: {e}')
            raise MCPManagerToolExecutionError(
                f'Tool execution failed: {e}') from e

    def __str__(self) -> str:
        return f"MCPTool(name='{self.name}', server='{self.tool_info.server_name}')"

    def __repr__(self) -> str:
        return self.__str__()


class MCPManager:
    """
    MCP (Model Context Protocol) Manager.

    This class manages multiple SingleServerMCPClient instances and provides
    a unified interface for tool management and execution.

    Key Features:
    - Manages multiple SingleServerMCPClient instances (1:1 with servers)
    - Provides unified tool interface across all servers
    - Thread-safe operations with proper resource management
    - Simple connection management: connect when needed, disconnect after use
    - Tool filtering and access control
    - Integrated with ModelScope authentication system
    """

    def __init__(
            self,
            mcp_config: Optional[Union[str, Dict[str, Any]]] = None,
            api_config: Optional[Dict[str, Any]] = None,
            tool_includes: Optional[List[Union[str, Dict[str,
                                                         List[str]]]]] = None,
            tool_excludes: Optional[List[Union[str, Dict[
                str, List[str]]]]] = DEFAULT_TOOL_EXCLUDES,
            warmup_connect: bool = True,  # Default enable warmup connection
            max_workers: int = 4,
            client_info: Optional[Any] = None,
            modelscope_token: Optional[str] = None,
            modelscope_base_url: Optional[str] = None,
            use_intl_site: Optional[bool] = False,
            connection_timeout: int = 60  # Connection timeout (seconds)
    ) -> None:
        """
        Initialize MCP Manager.

        Args:
            mcp_config: MCP configuration (file path or dict)
            api_config: API configuration for ModelScope Hub
            tool_includes: List of tools to include (filters)
            tool_excludes: List of tools to exclude (filters)
            warmup_connect: Whether to enable warmup connection (defaults to True)
            max_workers: Maximum number of worker threads
            client_info: Client information for MCP protocol
            modelscope_token: ModelScope SDK token (optional, will use saved token if not provided)
            modelscope_base_url: ModelScope API base URL (optional, defaults to official endpoint)
            use_intl_site: Whether to use international site (defaults to False for Chinese site)
            connection_timeout: Timeout for establishing connections (seconds)
        """
        # Initialize ModelScope authentication
        self._init_modelscope_auth(modelscope_token, modelscope_base_url)

        # Store configuration
        self.api_config = api_config or {}
        self.tool_includes = tool_includes or []
        self.tool_excludes = tool_excludes or []
        self.warmup_connect = warmup_connect
        self.max_workers = max_workers
        self.client_info = client_info or create_client_info(
            'modelscope-mcp-manager', '1.0.0')
        self.connection_timeout = connection_timeout

        # Initialize internal state
        self.servers: Dict[str, ServerInfo] = {}
        self.tools: Dict[str, MCPTool] = {}
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.loop_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.RLock()

        # Initialize MCP configuration (after servers is initialized)
        if mcp_config is not None or self.modelscope_token:
            self._load_configuration(mcp_config)

        # Start event loop in a separate thread
        self._start_loop()

        # Initialize tools (discover available tools without connecting)
        self._init_tools_sync()

        # Connection strategy
        if warmup_connect:
            # Warmup connection mode: quickly connect to get tool list, then disconnect
            if self.loop:
                asyncio.run_coroutine_threadsafe(self._warmup_connection(),
                                                 self.loop).result()
        else:
            # On-demand connection mode: connect only when needed
            logger.info('⚡ On-demand mode: connect only when tools are used')

    def _load_config_from_path(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file path with multiple search strategies.

        Args:
            config_path: Configuration file path (can be relative or absolute)

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If configuration file cannot be found
        """

        # 1. If absolute path is provided, try directly
        if os.path.isabs(config_path):
            if os.path.exists(config_path):
                return load_config(config_path)
            else:
                raise FileNotFoundError(
                    f'Configuration file not found at absolute path: {config_path}'
                )

        # 2. Try relative to current working directory
        current_dir = Path.cwd()
        candidate_paths = [
            current_dir / config_path,
            current_dir / 'mcp_config.json',
            current_dir / 'config' / 'mcp_config.json',
        ]

        # 3. Try relative to ModelScope package directory
        try:
            import modelscope
            modelscope_dir = Path(modelscope.__file__).parent
            candidate_paths.extend([
                modelscope_dir / 'hub' / 'modelscope_mcp' / config_path,
                modelscope_dir / 'hub' / 'modelscope_mcp' / 'mcp_config.json',
            ])
        except ImportError:
            pass

        # 4. Try relative to current script directory
        try:
            script_dir = Path(__file__).parent
            candidate_paths.extend([
                script_dir / config_path,
                script_dir / 'mcp_config.json',
            ])
        except NameError:
            pass

        # 5. Try environment variable specified path
        env_config_path = os.environ.get('MODELSCOPE_MCP_CONFIG')
        if env_config_path:
            candidate_paths.insert(0, Path(env_config_path))

        # Try all candidate paths
        for path in candidate_paths:
            if path.exists():
                logger.info(f'Loading MCP configuration from: {path}')
                return load_config(str(path))

        # If not found, provide detailed error information
        error_msg = f'Configuration file not found: {config_path}\n'
        error_msg += 'Searched in the following locations:\n'
        for path in candidate_paths:
            error_msg += f'  - {path}\n'
        error_msg += '\nTo fix this, you can:\n'
        error_msg += '1. Provide an absolute path to the configuration file\n'
        error_msg += '2. Set MODELSCOPE_MCP_CONFIG environment variable\n'
        error_msg += '3. Place mcp_config.json in the current working directory\n'
        error_msg += '4. Use a dictionary configuration instead of file path'

        raise FileNotFoundError(error_msg)

    def _init_modelscope_auth(self,
                              token: Optional[str],
                              base_url: Optional[str],
                              use_intl_site: bool = False) -> None:
        """
        Initialize ModelScope authentication.

        Args:
            token: ModelScope SDK token
            base_url: ModelScope API base URL
            use_intl_site: Whether to use international site
        """
        # Get token from parameter, environment variable, or saved credentials
        if token:
            self.modelscope_token = token
        else:
            # Try environment variable first
            self.modelscope_token = os.environ.get('MODELSCOPE_SDK_TOKEN')
            if not self.modelscope_token:
                # Try saved token from ModelScopeConfig
                self.modelscope_token = ModelScopeConfig.get_token()

        # Set base URL based on site preference
        if base_url:
            self.modelscope_base_url = base_url
        elif use_intl_site:
            self.modelscope_base_url = MODELSCOPE_INTL_API_BASE_URL
        else:
            self.modelscope_base_url = MODELSCOPE_API_BASE_URL

        # Store site preference
        self.use_intl_site = use_intl_site

        # Initialize HubApi for ModelScope operations with appropriate endpoint
        if use_intl_site:
            # Use international endpoint
            endpoint = get_endpoint(cn_site=False)
            self.hub_api = HubApi(endpoint=endpoint)
        else:
            # Use Chinese endpoint (default)
            self.hub_api = HubApi()

        # Login if token is available
        if self.modelscope_token:
            try:
                self.hub_api.login(access_token=self.modelscope_token)
                site_name = 'ModelScope International' if use_intl_site else 'ModelScope'
                logger.info(f'Successfully authenticated with {site_name}')
            except Exception as e:
                logger.warning(f'Failed to authenticate with ModelScope: {e}')
                self.modelscope_token = None
        else:
            site_name = 'ModelScope International' if use_intl_site else 'ModelScope'
            logger.info(
                f'No {site_name} token provided, operating in unauthenticated mode'
            )

    def get_modelscope_client(self):
        """
        Get ModelScope OpenAI-compatible client.

        Returns:
            OpenAI client configured for ModelScope API
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise MCPManagerError(
                'OpenAI library is required for ModelScope API access. '
                'Please install it with: pip install openai')

        if not self.modelscope_token:
            raise MCPManagerError(
                'ModelScope token is required for API access. '
                'Please provide a token or set MODELSCOPE_SDK_TOKEN environment variable.'
            )

        return OpenAI(
            api_key=self.modelscope_token, base_url=self.modelscope_base_url)

    def call_modelscope_model(self,
                              model: str,
                              messages: List[Dict[str, str]],
                              stream: bool = False,
                              **kwargs) -> Any:
        """
        Call ModelScope model using OpenAI-compatible interface.

        Args:
            model: Model ID (e.g., "Qwen/Qwen3-32B")
            messages: List of message dictionaries
            stream: Whether to stream the response
            **kwargs: Additional parameters for the API call

        Returns:
            API response from ModelScope
        """
        client = self.get_modelscope_client()

        try:
            # Convert messages to proper format for OpenAI client
            formatted_messages = []
            for msg in messages:
                if isinstance(msg,
                              dict) and 'role' in msg and 'content' in msg:
                    formatted_messages.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })
                else:
                    # Handle other message formats
                    formatted_messages.append(msg)

            response = client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                stream=stream,
                **kwargs)
            return response
        except Exception as e:
            logger.error(f'Failed to call ModelScope model {model}: {e}')
            raise MCPManagerError(f'ModelScope API call failed: {e}') from e

    def _load_configuration(
            self, mcp_config: Optional[Union[str, Dict[str, Any]]]) -> None:
        """
        Load and validate MCP configuration from local and/or remote sources.
        """
        # Load local configuration
        local_config = {}
        local_server_names = []
        if mcp_config is not None:
            if isinstance(mcp_config, str):
                try:
                    local_config = self._load_config_from_path(mcp_config)
                except Exception as e:
                    logger.warning(
                        f'Failed to load local MCP config from file: {e}')
                    local_config = {}
            else:
                local_config = mcp_config
        else:
            logger.info('No local MCP configuration provided')

        # Load remote configuration from ModelScope Hub (if token is available)
        remote_config = {}
        remote_server_names = []
        if self.modelscope_token:
            try:
                # Create HubApi instance
                base_api = HubApi()
                mcp_api = McpApi(base_api)

                # Get operational servers with user's token
                operational_result = mcp_api.get_mcp_server_operational(
                    token=self.modelscope_token, convert_to_mcp_config=True)
                remote_config = operational_result
            except Exception as e:
                logger.warning(f'Failed to load remote MCP config: {e}')
                remote_config = {'mcpServers': {}}
        else:
            remote_config = {'mcpServers': {}}

        # Merge configurations based on what's available
        if local_config and remote_config:
            # Both configs exist, merge them
            if 'mcpServers' in local_config and 'mcpServers' in remote_config:
                merged_servers = {
                    **local_config['mcpServers'],
                    **remote_config['mcpServers']
                }
                self.mcp_config = {'mcpServers': merged_servers}
                local_server_names = list(local_config['mcpServers'].keys())
                remote_server_names = list(remote_config['mcpServers'].keys())
            elif 'mcpServers' in local_config:
                self.mcp_config = local_config
                local_server_names = list(local_config['mcpServers'].keys())
            elif 'mcpServers' in remote_config:
                self.mcp_config = remote_config
                remote_server_names = list(remote_config['mcpServers'].keys())
            else:
                self.mcp_config = {'mcpServers': {}}
        elif local_config:
            self.mcp_config = local_config
            local_server_names = list(
                local_config.get('mcpServers', {}).keys())
        elif remote_config:
            self.mcp_config = remote_config
            remote_server_names = list(
                remote_config.get('mcpServers', {}).keys())
        else:
            self.mcp_config = {'mcpServers': {}}
            logger.warning('No MCP configuration loaded from any source')

        # Extract server configurations
        mcp_servers = self.mcp_config.get('mcpServers', {})
        if not mcp_servers:
            logger.warning('No MCP servers found in configuration')
            mcp_servers = self.mcp_config  # Fallback to direct server config

        # Validate server configurations
        validate_mcp_config(mcp_servers)

        # Initialize server information
        for server_name, server_config in mcp_servers.items():
            self.servers[server_name] = ServerInfo(
                name=server_name, config=server_config)
        # Save local and remote server names for later printing
        self._local_server_names = local_server_names
        self._remote_server_names = remote_server_names

    def _start_loop(self) -> None:
        """Start the event loop in a separate thread."""
        # Create a new event loop for this thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Start the event loop in a separate thread
        self.loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.loop_thread.start()

    def _run_loop(self) -> None:
        """Run the event loop."""
        if self.loop:
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

    def _init_tools_sync(self) -> None:
        """Initialize tools synchronously."""
        if self.loop:
            future = asyncio.run_coroutine_threadsafe(self._init_tools_async(),
                                                      self.loop)
            future.result()

    async def _init_tools_async(self) -> None:
        """Initialize tools asynchronously."""
        try:
            # Register available tools (may connect servers if auto_connect=True)
            await self._register_available_tools()

            # Apply filters
            self._apply_tool_filters()

            # Count connection results
            if self.warmup_connect:
                logger.debug(
                    f'🔄 Registered {len(self.servers)} servers for warmup connection'
                )
            else:
                logger.debug(
                    f'🔄 Registered {len(self.servers)} servers for on-demand connection'
                )

        except Exception as e:
            logger.error(f'Failed to initialize tools: {e}')
            raise MCPManagerInitializationError(
                f'Tool initialization failed: {e}') from e

    async def _connect_server(self, server_name: str,
                              server_info: ServerInfo) -> MCPClient:
        """Connect to a specific MCP server and return the client."""
        try:
            # Create client info
            client_info = create_client_info(
                f'{self.client_info.name}-{server_name}',
                self.client_info.version)

            # Create MCPClient
            client = MCPClient(
                server_name=server_name,
                server_config=server_info.config,
                client_info=client_info)

            # Connect to server
            logger.info(f'🔗 Connecting to server: {server_name}')
            await client.connect()
            logger.info(f'✅ Connected to server: {server_name}')
            return client

        except Exception as e:
            logger.error(f'❌ Failed to connect to server {server_name}: {e}')
            server_info.error_count += 1
            raise

    async def _warmup_connection(self) -> None:
        """
        Warm up: quickly connect all servers, get tool list, then disconnect.
        """
        # Create task list
        warmup_tasks = []
        for server_name, server_info in self.servers.items():
            task = asyncio.create_task(
                self._warmup_server(server_name, server_info))
            warmup_tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*warmup_tasks, return_exceptions=True)

        total_tools = len(self.tools)
        total_servers = len(self.servers)
        local_names = getattr(self, '_local_server_names', [])
        remote_names = getattr(self, '_remote_server_names', [])
        print(
            f"[MCP Stats] Loaded {len(local_names)} local servers: {', '.join(local_names) if local_names else 'None'}"
        )
        remote_servers_str = ', '.join(
            remote_names) if remote_names else 'None'
        print(
            f'[MCP Stats] Loaded {len(remote_names)} modelscope remote servers: {remote_servers_str}'
        )
        print(f'[MCP Stats] Total MCP servers: {total_servers}')
        print(f'[MCP Stats] Total tools discovered: {total_tools}')

    async def _discover_server_tools(self, server_name: str,
                                     server_info: ServerInfo) -> int:
        """
        Unified tool discovery method: connect to server and discover tools.

        Args:
            server_name: Server name
            server_info: Server information

        Returns:
            Number of discovered tools
        """
        try:
            # Connect to server
            client = await self._connect_server(server_name, server_info)

            # Get tool list
            server_tools = await client.get_tools()
            tool_count = len(server_tools)

            # Register tools
            for tool in server_tools:
                tool_info = ToolInfo(
                    name=format_mcp_tool_name(server_name, tool.name),
                    description=getattr(tool, 'description', tool.name),
                    parameters=getattr(tool, 'inputSchema', {}),
                    server_name=server_name,
                    tool_name=tool.name)
                server_info.tools.append(tool_info)
                self.tools[tool_info.name] = MCPTool(tool_info, self)

            # Disconnect
            await client.disconnect()
            logger.info(
                f'✅ Server {server_name}: {tool_count} tools discovered')
            return tool_count

        except Exception as e:
            logger.warning(
                f'Failed to discover tools from server {server_name}: {e}')
            return 0

    async def _warmup_server(self, server_name: str,
                             server_info: ServerInfo) -> None:
        """Warm up single server: connect, get tools, disconnect."""
        tool_count = await self._discover_server_tools(server_name,
                                                       server_info)
        logger.debug(
            f'✅ Server {server_name} warmed up: {tool_count} tools discovered')

    async def _register_available_tools(self) -> None:
        """Register available tools, but don't connect to servers immediately."""
        # Only register server information, tools will be discovered dynamically on first use
        logger.debug('Tools will be discovered on-demand')
        for server_name, server_info in self.servers.items():
            logger.debug(
                f'Registered server: {server_name} (connection pending)')

    def _apply_tool_filters(self) -> None:
        """Apply tool inclusion and exclusion filters."""
        filtered_tools = {}

        for tool_name, tool in self.tools.items():
            server_name, api_name = parse_mcp_tool_name(tool_name)

            # Check includes
            if self.tool_includes and not self._matches_filter(
                    self.tool_includes, server_name, api_name):
                continue

            # Check excludes
            if self.tool_excludes and self._matches_filter(
                    self.tool_excludes, server_name, api_name):
                continue

            filtered_tools[tool_name] = tool

        self.tools = filtered_tools

    def _matches_filter(self, filter_list: List[Union[str, Dict[str,
                                                                List[str]]]],
                        server_name: str, api_name: Optional[str]) -> bool:
        """Check if a tool matches the filter criteria."""
        for filter_item in filter_list:
            if isinstance(filter_item, str):
                if filter_item == server_name:
                    return True
            elif isinstance(filter_item, dict):
                if server_name in filter_item:
                    api_list = filter_item[server_name]
                    if not api_list or api_name in api_list:
                        return True
        return False

    async def _call_tool_async(self, server_name: str, tool_name: str,
                               tool_args: Dict[str, Any]) -> str:
        """Call a tool asynchronously with simple connect-use-disconnect pattern."""
        server_info = self.servers.get(server_name)
        if not server_info:
            raise MCPManagerConnectionError(f'Server {server_name} not found')

        client = None
        try:
            # 1. Connect to server
            logger.info(
                f'🔗 Connecting to server {server_name} for tool {tool_name}')
            client = await self._connect_server(server_name, server_info)

            # 2. Execute tool call (tool_name is the original tool name on server, validated during warmup)
            logger.info(
                f'⚡ Executing tool {tool_name} on server {server_name}')
            result = await client.call_tool(tool_name, tool_args)
            logger.info(
                f'✅ Tool {tool_name} executed successfully on server {server_name}'
            )

            # 3. Disconnect
            logger.info(f'🔌 Disconnecting from server {server_name}')
            await client.disconnect()

            return result

        except Exception as e:
            # Ensure disconnection
            if client:
                try:
                    await client.disconnect()
                except Exception:
                    pass

            logger.error(
                f'❌ Failed to execute tool {tool_name} on server {server_name}: {e}'
            )
            raise MCPManagerToolExecutionError(
                f'Tool execution failed: {e}') from e

    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get all available tools with detailed information including parameters.

        Returns:
            List of tools with detailed information (name, description, parameters, etc.)
        """
        with self._lock:
            detailed_tools = []
            for tool in self.tools.values():
                tool_info = {
                    'name': tool.name,
                    'description': tool.description,
                    'server_name': tool.tool_info.server_name,
                    'tool_name': tool.tool_info.tool_name,
                    'parameters': tool.parameters,
                    'usage_count': tool.tool_info.usage_count,
                    'last_used': tool.tool_info.last_used
                }
                detailed_tools.append(tool_info)
            return detailed_tools

    def get_tool_by_name(
            self,
            tool_name: str,
            server_name: Optional[str] = None) -> Optional[MCPTool]:
        """
        Get a tool by name.

        Args:
            tool_name: Name of the tool (can be full name like "amap-maps---maps_weather" or just "maps_weather")
            server_name: Optional server name to search within (if provided, tool_name should be the original tool name)

        Returns:
            Tool if found, None otherwise
        """
        with self._lock:
            # If server name is provided, search by server + tool name
            if server_name:
                for tool in self.tools.values():
                    if tool.tool_info.server_name == server_name and tool.tool_info.tool_name == tool_name:
                        return tool
                return None

            # Otherwise search by full tool name (backward compatibility)
            return self.tools.get(tool_name)

    def get_tool_by_server_and_name(self, server_name: str,
                                    tool_name: str) -> Optional[MCPTool]:
        """
        Get a tool by server name and original tool name.

        Args:
            server_name: Name of the server
            tool_name: Original tool name on the server

        Returns:
            Tool if found, None otherwise
        """
        return self.get_tool_by_name(tool_name, server_name)

    def get_tools_by_server(self, server_name: str) -> List[MCPTool]:
        """
        Get all tools from a specific server.

        Args:
            server_name: Name of the server

        Returns:
            List of tools from the server
        """
        with self._lock:
            return [
                tool for tool in self.tools.values()
                if tool.tool_info.server_name == server_name
            ]

    def list_available_servers(self) -> List[str]:
        """
        List all available servers.

        Returns:
            List of server names
        """
        with self._lock:
            return list(self.servers.keys())

    def get_server_status(self, server_name: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific server.

        Args:
            server_name: Name of the server

        Returns:
            Server status information
        """
        with self._lock:
            if server_name not in self.servers:
                return None

            server_info = self.servers[server_name]
            return {
                'name': server_info.name,
                'error_count': server_info.error_count,
                'tool_count': len(server_info.tools),
                'is_connected': len(server_info.tools) > 0,
                'is_reconnecting': False
            }

    def get_tool_statistics(self) -> Dict[str, Any]:
        """
        Get tool usage statistics.

        Returns:
            Tool statistics
        """
        with self._lock:
            total_tools = len(self.tools)
            server_stats = {}

            for server_name, server_info in self.servers.items():
                server_stats[server_name] = {
                    'tool_count': len(server_info.tools),
                    'error_count': server_info.error_count
                }

            # Get most used tools
            sorted_tools = sorted(
                self.tools.values(),
                key=lambda t: t.tool_info.usage_count,
                reverse=True)

            most_used = [{
                'name': tool.name,
                'usage_count': tool.tool_info.usage_count,
                'last_used': tool.tool_info.last_used
            } for tool in sorted_tools[:5]]

            return {
                'total_tools': total_tools,
                'servers': server_stats,
                'most_used_tools': most_used
            }

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """
        Get all available tools in OpenAI-compatible format.

        Returns:
            List of tools in OpenAI function calling format
        """
        with self._lock:
            tools = []
            for tool in self.tools.values():
                openai_tool = {
                    'type': 'function',
                    'function': {
                        'name': tool.name,
                        'description': tool.tool_info.description,
                        'parameters': tool.tool_info.parameters
                    }
                }
                tools.append(openai_tool)
            return tools

    async def discover_tools_async(self) -> Dict[str, List[str]]:
        """
        Discover tools from all servers asynchronously.

        Returns:
            Dictionary mapping server names to lists of available tool names
        """
        discovered_tools = {}

        for server_name, server_info in self.servers.items():
            try:
                # Use unified tool discovery method
                await self._discover_server_tools(server_name, server_info)
                tool_names = [tool.name for tool in server_info.tools]
                discovered_tools[server_name] = tool_names

            except Exception as e:
                logger.error(
                    f'Failed to discover tools from server {server_name}: {e}')
                discovered_tools[server_name] = []

        return discovered_tools

    def discover_tools(self) -> Dict[str, List[str]]:
        """
        Discover tools from all servers synchronously.

        Returns:
            Dictionary mapping server names to lists of available tool names
        """
        if self.loop:
            future = asyncio.run_coroutine_threadsafe(
                self.discover_tools_async(), self.loop)
            return future.result()
        else:
            logger.error('Event loop not available')
            return {}

    def get_tools_summary(self) -> Dict[str, Any]:
        """
        Get tool summary information for LLM to understand available tools.

        Returns:
            Dictionary containing tool summary information
        """
        with self._lock:
            summary = {
                'total_tools': len(self.tools),
                'servers': {},
                'tools_by_category': {},
                'connection_mode':
                'warmup' if self.warmup_connect else 'on_demand'
            }

            # Group by server
            for server_name, server_info in self.servers.items():
                server_tools = [
                    tool for tool in self.tools.values()
                    if tool.tool_info.server_name == server_name
                ]
                summary['servers'][server_name] = {
                    'tool_count': len(server_tools),
                    'tools': [tool.name for tool in server_tools]
                }

            # Group by tool name prefix (simple categorization)
            for tool in self.tools.values():
                category = tool.name.split(
                    '_')[0] if '_' in tool.name else 'general'
                if category not in summary['tools_by_category']:
                    summary['tools_by_category'][category] = []
                summary['tools_by_category'][category].append({
                    'name':
                    tool.name,
                    'description':
                    tool.tool_info.description,
                    'server':
                    tool.tool_info.server_name
                })

            return summary

    def list_tools_for_llm(self) -> str:
        """
        Generate tool list description suitable for LLM reading.

        Returns:
            Formatted tool list string
        """
        summary = self.get_tools_summary()

        result = f"Total available tools: {summary['total_tools']}\n"
        result += f"Connection mode: {summary['connection_mode']}\n\n"

        result += 'Grouped by server:\n'
        for server_name, info in summary['servers'].items():
            result += f"  {server_name}: {info['tool_count']} tools\n"
            for tool_name in info['tools'][:5]:  # Only show first 5
                result += f'    - {tool_name}\n'
            if len(info['tools']) > 5:
                result += f"    ... and {len(info['tools']) - 5} more tools\n"

        result += '\nGrouped by category:\n'
        for category, tools in summary['tools_by_category'].items():
            result += f'  {category}: {len(tools)} tools\n'
            for tool in tools[:3]:  # Only show first 3
                result += f"    - {tool['name']}: {tool['description'][:50]}...\n"
            if len(tools) > 3:
                result += f'    ... and {len(tools) - 3} more tools\n'

        return result

    def query_service_registry(
            self, keywords: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Query service registry and return matching service information based on keywords.

        Args:
            keywords: Query keywords, can be string or list of keywords

        Returns:
            List of matching services with service metadata
        """
        with self._lock:
            if isinstance(keywords, str):
                keywords = [keywords.lower()]
            else:
                keywords = [kw.lower() for kw in keywords]

            matched_services = []

            for tool in self.tools.values():
                # Check if tool name, description, server name contains keywords
                tool_name = tool.name.lower()
                description = tool.description.lower()
                server_name = tool.tool_info.server_name.lower()

                # Check if matches any keyword
                is_match = any(
                    kw in tool_name or kw in description or kw in server_name
                    for kw in keywords)

                if is_match:
                    service_info = {
                        'service_id': tool.name,
                        'name': tool.name,
                        'description': tool.description,
                        'server': tool.tool_info.server_name,
                        'parameters': tool.parameters,
                        'call_method': f'MCP tool call: {tool.name}'
                    }
                    matched_services.append(service_info)

            return matched_services

    def get_service_metadata(self,
                             service_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed metadata for specified service.

        Args:
            service_id: Service ID (tool name)

        Returns:
            Service metadata dictionary, returns None if service doesn't exist
        """
        with self._lock:
            tool = self.tools.get(service_id)
            if not tool:
                return None

            return {
                'service_id': tool.name,
                'name': tool.name,
                'description': tool.description,
                'server': tool.tool_info.server_name,
                'parameters': tool.parameters,
                'call_method': f'MCP tool call: {tool.name}'
            }

    def get_service_brief_summary(self) -> str:
        """
        Generate service brief information for embedding in system prompt.

        Returns:
            Formatted service brief information string
        """
        with self._lock:
            if not self.tools:
                return 'No available services'

            # Generate brief information grouped by server
            server_groups = {}
            for tool in self.tools.values():
                server_name = tool.tool_info.server_name
                if server_name not in server_groups:
                    server_groups[server_name] = []
                server_groups[server_name].append(tool)

            # Generate brief information
            brief_info = []
            for server_name, tools in server_groups.items():
                # Generate brief description for each tool
                for tool in tools:
                    # Extract core functionality description (first 50 characters)
                    description = tool.description[:50] + '...' if len(
                        tool.description) > 50 else tool.description
                    brief_info.append(f'  - {tool.name}: {description}')
                brief_info.append('')

            return '\n'.join(brief_info)

    def get_service_brief_for_prompt(self) -> str:
        """
        Generate service brief information suitable for system prompt.

        Returns:
            Formatted service brief information with usage instructions
        """
        brief_summary = self.get_service_brief_summary()

        return f"""## Available Services

{brief_summary}

## Usage Rules
- First determine if user question requires service calls;
- If needed, first call query_service_registry to query related services, keywords parameter can be string or string list, such as ["weather", "train"] or "weather";
- After finding matching services, if parameter details are needed, call get_service_metadata, service_id parameter must be complete service ID, such as "amap-maps---maps_weather";
- After confirming service information, generate service call instructions.
- **Direct call**: For clear services (such as weather, date), can call directly without querying service registry first
- **Call format**: Wrap with <function_call>, such as:
  <function_call>
  {{
      "name": "amap-maps---maps_weather",
      "parameters": {{"city": "Hangzhou"}}
  }}
  </function_call>
"""  # noqa: E501

    def shutdown(self) -> None:
        """Shutdown the manager and clean up resources."""
        try:
            # Shutdown executor
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)

            # Stop event loop
            if self.loop and self.loop_thread:
                try:
                    self.loop.call_soon_threadsafe(self.loop.stop)
                    self.loop_thread.join(timeout=5)
                except Exception as e:
                    logger.warning(f'Error stopping event loop: {e}')

            # Silently complete shutdown, don't print logs

        except Exception as e:
            logger.error(f'Error during shutdown: {e}')

    def __enter__(self) -> 'MCPManager':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.shutdown()


# Convenience function for creating manager
def create_mcp_manager(mcp_config: Optional[Union[str, Dict[str, Any]]] = None,
                       api_config: Optional[Dict[str, Any]] = None,
                       modelscope_token: Optional[str] = None,
                       use_intl_site: bool = False,
                       warmup_connect: bool = True,
                       max_workers: int = 4,
                       **kwargs) -> MCPManager:
    """
    Create an MCP Manager with the given configuration.

    Args:
        mcp_config: MCP configuration (optional, will load from remote if token provided)
        api_config: API configuration
        modelscope_token: ModelScope token for accessing remote MCP servers
        use_intl_site: Whether to use international site (defaults to False for Chinese site)
        warmup_connect: Whether to enable warmup connection (defaults to True)
        max_workers: Maximum number of worker threads
        **kwargs: Additional arguments for MCPManager

    Returns:
        Configured MCP Manager
    """
    return MCPManager(
        mcp_config=mcp_config,
        api_config=api_config,
        modelscope_token=modelscope_token,
        use_intl_site=use_intl_site,
        warmup_connect=warmup_connect,
        max_workers=max_workers,
        **kwargs)
