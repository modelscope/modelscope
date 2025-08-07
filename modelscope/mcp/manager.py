# Copyright (c) Alibaba, Inc. and its affiliates.
"""
MCP Manager - Unified interface for managing multiple MCP clients

This module provides a high-level abstraction that hides the complexity of managing
multiple MCP clients, their lifecycle, and tool routing. Users only need to call
mcp.call_tools(tool_name, tool_args) to execute any tool on any server.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union

import json

from modelscope.hub.api import HubApi
from modelscope.utils.logger import get_logger
from .api import MCPApi
from .client import MCPClient
from .utils import MCPRequestAdapter

logger = get_logger(__name__)


class MCP:
    """
    Unified MCP manager that provides a simple interface for multi-server tool execution

    This class manages multiple MCP (Model Context Protocol) servers, handles their lifecycle,
    discovers available tools, and provides a unified interface for tool execution. It supports
    intelligent connection strategies, concurrent execution, and framework-agnostic requests.

    Features:
    - Automatic client lifecycle management
    - Multi-server configuration support (local files + ModelScope Hub)
    - Tool discovery and routing with conflict resolution
    - Framework-agnostic request/response handling (OpenAI, LangChain, Transformers, etc.)
    - Concurrent tool execution support
    - Intelligent connection strategies based on transport type
    - Automatic deduplication and error handling

    Connection Strategies:
    - STDIO: Always ephemeral (create-connect-use-disconnect) - best for local processes
    - SSE: Persistent connection reuse for high-frequency servers
    - Streamable HTTP: Ephemeral by default (optimizable based on usage patterns)

    Basic Usage:
        >>> # Initialize with ModelScope token (warmup runs automatically)
        >>> mcp = MCP(token="ms-your-token-here")

        >>> # Single tool call
        >>> result = await mcp.call_tools({"name": "server1:search", "arguments": {"query": "test"}})
        >>> print(result)

        >>> # Multiple concurrent tool calls
        >>> results = await mcp.call_tools([
        ...     {"name": "server1:search", "arguments": {"query": "python"}},
        ...     {"name": "server2:translate", "arguments": {"text": "hello", "target": "zh"}}
        ... ])
        >>> for result in results:
        ...     print(f"{result['tool_name']}: {result['success']}")

        >>> # List available tools
        >>> print(mcp.list_tools())

        >>> # Get statistics
        >>> stats = mcp.get_stats()
        >>> print(f"Total tools: {stats['total_tools']}")

        >>> # Cleanup when done
        >>> mcp.shutdown()

    Advanced Configuration:
        >>> # Use local configuration file
        >>> mcp = MCP(
        ...     token="ms-your-token-here",
        ...     config="/path/to/mcp-config.json",
        ...     config_mode="local"
        ... )

        >>> # Merge local and remote configurations
        >>> mcp = MCP(
        ...     token="ms-your-token-here",
        ...     config={"mcpServers": {"local_server": {...}}},
        ...     config_mode="merge"
        ... )

        >>> # Remote-only mode (ignore local config)
        >>> mcp = MCP(
        ...     token="ms-your-token-here",
        ...     config_mode="remote"
        ... )

    Framework Integration Examples:
        >>> # OpenAI-style function call
        >>> openai_call = {
        ...     "name": "server1:calculator",
        ...     "arguments": '{"operation": "add", "a": 5, "b": 3}'
        ... }
        >>> result = await mcp.call_tools(openai_call)

        >>> # Transformers-style function call
        >>> transformers_call = {
        ...     "function": {
        ...         "name": "server1:search",
        ...         "parameters": {"query": "machine learning"}
        ...     }
        ... }
        >>> result = await mcp.call_tools(transformers_call)

    Error Handling:
        >>> try:
        ...     result = await mcp.call_tools({"name": "invalid:tool", "arguments": {}})
        ... except ValueError as e:
        ...     print(f"Tool not found: {e}")
        ... except Exception as e:
        ...     print(f"Execution failed: {e}")

    Authentication:
        >>> # Get your ModelScope token from: https://modelscope.cn/my/myaccesstoken
        >>> token = "ms-your-32-character-token-here"
        >>> mcp = MCP(token=token)

        >>> # For enterprise users with specific endpoints
        >>> mcp = MCP(
        ...     token=token,
        ...     combination="enterprise",  # or specific server combination
        ... )

    Configuration File Format:
        >>> # Local MCP configuration file (JSON)
        >>> config = {
        ...     "mcpServers": {
        ...         "my_local_server": {
        ...             "type": "stdio",
        ...             "command": "/path/to/server",
        ...             "args": ["--port", "8080"]
        ...         },
        ...         "my_sse_server": {
        ...             "type": "sse",
        ...             "url": "https://example.com/mcp/sse"
        ...         }
        ...     }
        ... }
        >>> mcp = MCP(token="your-token", config=config, config_mode="merge")

    Performance Tips:
        >>> # For high-frequency tool calls, SSE servers use persistent connections
        >>> # STDIO servers always use ephemeral connections (safer for local processes)
        >>> # Check connection strategies:
        >>> stats = mcp.get_stats()
        >>> print(f"Connection strategies: {stats['connection_strategies']}")
        >>> print(f"Active persistent connections: {stats['active_persistent_connections']}")

    Tool Naming Convention:
        >>> # Tools are automatically prefixed with server name to avoid conflicts
        >>> # Format: "servername:toolname"
        >>> # Original tool "mcp" on server "ModelScope" becomes "ModelScope:mcp"
        >>> result = await mcp.call_tools({"name": "ModelScope:mcp", "arguments": {"q": "python"}})

    Attributes:
        token (str): ModelScope authentication token
        combination (str): Server combination filter (default: "ALL")
        config (Union[str, Dict]): Additional configuration source
        config_mode (str): Configuration loading mode ("local", "merge", "remote")
        server_configs (Dict): Loaded server configurations
        tool_list (List): Discovered tools from all servers
        tool_to_server (Dict): Tool name to server mapping
        clients (Dict): Active persistent client connections
        stats (Dict): Execution statistics and metrics

    Note:
        - Initialization includes automatic warmup that discovers all available tools
        - Warmup is blocking and must complete before the instance is ready
        - All tool execution is async and supports concurrent operations
        - Persistent connections are automatically managed and recovered on failure
        - Tool names must include server prefix to avoid conflicts between servers
    """

    def __init__(
        self,
        token: str,
        combination: Optional[str] = 'ALL',
        config: Optional[Union[str, Dict[str, Any]]] = None,
        config_mode: Optional[str] = 'remote',
    ):
        """
        Initialize MCP Manager with automatic tool discovery and connection strategy optimization

        This method performs complete initialization including server configuration loading,
        tool discovery, and connection strategy determination. It blocks until warmup is complete
        and the instance is ready for tool execution.

        Args:
            token (str): ModelScope authentication token (required)
                Get yours from: https://modelscope.cn/my/myaccesstoken
            combination (str, optional): Server combination filter (default: "ALL")
                - "ALL": Load all available servers
            config (Union[str, Dict], optional): Additional local configuration
                - str: Path to JSON configuration file
                - Dict: Configuration dictionary with "mcpServers" key
                - None: No local configuration (default)
            config_mode (str, optional): Configuration loading strategy (default: "remote")
                - "local": Prefer local config, fallback to ModelScope Hub if local fails
                - "merge": Load both sources, merge and deduplicate (local overrides remote)
                - "remote": Only use ModelScope Hub, ignore local config completely

        Raises:
            ValueError: If no valid servers can be loaded from any source
            ConnectionError: If unable to connect to ModelScope Hub API
            FileNotFoundError: If config file path is invalid (in local/merge modes)
            JSONDecodeError: If config file contains invalid JSON

        Examples:
            >>> # Basic initialization with ModelScope token
            >>> mcp = MCP(token="ms-your-token-here")

            >>> # Use local configuration file
            >>> mcp = MCP(
            ...     token="ms-your-token-here",
            ...     config="/path/to/mcp-config.json",
            ...     config_mode="local"
            ... )

            >>> # Merge local and remote configurations
            >>> local_config = {
            ...     "mcpServers": {
            ...         "local_calculator": {
            ...             "type": "stdio",
            ...             "command": "/usr/local/bin/calc-server"
            ...         }
            ...     }
            ... }
            >>> mcp = MCP(
            ...     token="ms-your-token-here",
            ...     config=local_config,
            ...     config_mode="merge"
            ... )

            >>> # Enterprise servers only
            >>> mcp = MCP(
            ...     token="ms-your-token-here",
            ...     combination="enterprise",
            ...     config_mode="remote"
            ... )

        Initialization Process:
            1. Validate authentication token
            2. Load server configurations based on config_mode
            3. Deduplicate servers by name and URL
            4. Determine optimal connection strategy for each server
            5. Discover available tools from all servers (ephemeral connections)
            6. Build tool-to-server routing table
            7. Generate tool prompt for LLM consumption

        Connection Strategy Assignment:
            - STDIO servers: Always ephemeral (safer for local processes)
            - SSE servers: Persistent connections (optimized for high-frequency calls)
            - Streamable HTTP: Ephemeral by default (future optimization possible)

        Post-Initialization State:
            >>> # After successful initialization, you can:
            >>> print(f"Loaded {mcp.stats['total_servers']} servers")
            >>> print(f"Discovered {mcp.stats['total_tools']} tools")
            >>> print(mcp.list_tools())  # View all available tools
            >>> stats = mcp.get_stats()  # Get detailed statistics

        Note:
            - Initialization is synchronous and blocking until warmup completes
            - All servers are tested for connectivity during warmup
            - Failed server connections are logged but don't prevent initialization
            - Tool names are automatically prefixed with server names to avoid conflicts
            - The instance is immediately ready for tool execution after __init__ returns
        """
        self.token = token
        self.combination = combination
        self.config = config
        self.config_mode = config_mode

        # Core data structures
        self.server_configs: Dict[str, Dict[str,
                                            Any]] = {}  # server_id -> config
        self.tool_list: List[Dict[str, Any]] = []  # All available tools
        self.tool_to_server: Dict[str, str] = {}  # tool_name -> server_id
        self.clients: Dict[str, MCPClient] = {
        }  # Client instances for persistent connections
        self.tool_prompt: str = ''  # Generated prompt for LLM

        # Connection strategy state
        self._connection_strategies: Dict[str,
                                          str] = {}  # server_name -> strategy
        self._client_usage_count: Dict[str,
                                       int] = {}  # server_name -> usage count
        self._last_used_time: Dict[str, float] = {
        }  # server_name -> last used timestamp

        # Statistics
        self.stats = {
            'total_servers': 0,
            'total_tools': 0,
            'successful_connections': 0,
            'failed_connections': 0,
            'tool_calls': 0,
            'persistent_connections': 0,
            'ephemeral_connections': 0
        }

        # Initialize API client
        base_api = HubApi()
        self.api = MCPApi(base_api)

        # Run warmup to populate self information
        asyncio.run(self._warmup())

    def _determine_connection_strategy(self, server_name: str,
                                       server_config: Dict[str, Any]) -> str:
        """
        Determine the optimal connection strategy for a server based on its transport type

        Args:
            server_name: Name of the server
            server_config: Server configuration

        Returns:
            str: Connection strategy ('ephemeral' or 'persistent')
        """
        transport_type = server_config.get('type', 'unknown')

        if transport_type == 'stdio':
            # STDIO: Always use ephemeral (即插即用) - best for local processes
            strategy = 'ephemeral'
            logger.debug(
                f'Server {server_name}: STDIO transport -> ephemeral strategy')

        elif transport_type == 'sse':
            # SSE: Use persistent strategy for better performance
            strategy = 'persistent'
            logger.debug(
                f'Server {server_name}: SSE transport -> persistent strategy')

        elif transport_type == 'streamable_http':
            # Streamable HTTP: Use ephemeral by default (can be optimized later based on usage)
            strategy = 'ephemeral'
            logger.debug(
                f'Server {server_name}: Streamable HTTP transport -> ephemeral strategy'
            )

        else:
            # Unknown transport: Default to ephemeral for safety
            strategy = 'ephemeral'
            logger.debug(
                f"Server {server_name}: Unknown transport '{transport_type}' -> ephemeral strategy"
            )

        self._connection_strategies[server_name] = strategy
        return strategy

    async def _warmup(self) -> None:
        """
        Warmup process: Load configurations and discover all available tools

        This method:
        1. Loads server configurations from ModelScope Hub or local config
        2. Connects to all servers to discover available tools (using ephemeral connections)
        3. Builds tool-to-server mapping
        4. Determines connection strategies for each server
        5. Generates tool prompt for LLM consumption
        """
        logger.info('Starting MCP warmup process...')

        try:
            # Step 1: Load server configurations
            await self._load_server_configs()

            # Step 2: Determine connection strategies for all servers
            for server_name, server_config in self.server_configs.items():
                self._determine_connection_strategy(server_name, server_config)

            # Step 3: Discover tools from all servers (using ephemeral connections during warmup)
            await self._discover_tools()

            # Step 4: Generate tool prompt
            self._generate_tools_prompt()

            logger.info(f'MCP warmup completed successfully. '
                        f"Loaded {self.stats['total_servers']} servers "
                        f"with {self.stats['total_tools']} tools total.")

        except Exception as e:
            logger.error(f'MCP warmup failed: {e}')
            raise

    async def _load_server_configs(self) -> None:
        """Load server configurations from multiple sources with deduplication"""

        if self.config_mode == 'local':
            await self._load_configs_local()
        elif self.config_mode == 'merge':
            await self._load_configs_merge()
        elif self.config_mode == 'remote':
            await self._load_configs_remote()
        else:
            raise ValueError(f'Unsupported config_mode: {self.config_mode}. '
                             "Use 'local', 'merge', or 'remote'.")

        # Detect URL-based duplicates
        self._deduplicate_by_url()

        total_servers = len(self.server_configs)
        self.stats['total_servers'] = total_servers
        logger.info(f'Final server count after deduplication: {total_servers}')

    async def _load_configs_local(self) -> None:
        """Mode 1: Use local config if available, fallback to ModelScope Hub"""

        # Try to load local config first
        local_loaded = False
        local_servers_count = 0

        if self.config:
            try:
                local_configs = self._parse_local_config(self.config)
                local_servers = local_configs.get('mcpServers', {})

                if local_servers:
                    for server_name, server_config in local_servers.items():
                        self.server_configs[server_name] = server_config

                    local_servers_count = len(local_servers)
                    local_loaded = True
                    logger.info(
                        f'✅ Using local config: {local_servers_count} servers loaded'
                    )
                else:
                    logger.warning(
                        '⚠️ Local config exists but contains no servers')

            except Exception as e:
                logger.warning(f'❌ Failed to load local config: {e}')

        # If no local config or loading failed, try ModelScope Hub
        if not local_loaded:
            try:
                hub_configs = self.api.list_operational_mcp_servers(
                    token=self.token, endpoint=None)

                mcp_servers = hub_configs.get('mcpServers', {})
                for server_name, server_config in mcp_servers.items():
                    self.server_configs[server_name] = server_config

                hub_servers_count = len(mcp_servers)
                logger.info(
                    f'✅ Using ModelScope Hub: {hub_servers_count} servers loaded'
                )

            except Exception as e:
                logger.error(f'❌ Failed to load from ModelScope Hub: {e}')

        # Ensure at least one source succeeded
        if not self.server_configs:
            raise ValueError(
                'No server configurations could be loaded from any source')

    async def _load_configs_merge(self) -> None:
        """Mode 2: Load both local and ModelScope Hub configs, merge and deduplicate"""

        hub_servers_count = 0
        local_servers_count = 0
        overridden_servers = []

        # Load from ModelScope Hub first
        try:
            hub_configs = self.api.list_operational_mcp_servers(
                token=self.token, endpoint=None)

            mcp_servers = hub_configs.get('mcpServers', {})
            for server_name, server_config in mcp_servers.items():
                self.server_configs[server_name] = server_config

            hub_servers_count = len(mcp_servers)
            logger.info(
                f'📡 Loaded {hub_servers_count} servers from ModelScope Hub')

        except Exception as e:
            logger.warning(
                f'❌ Failed to load servers from ModelScope Hub: {e}')

        # Load from local config if provided
        if self.config:
            try:
                local_configs = self._parse_local_config(self.config)
                local_servers = local_configs.get('mcpServers', {})

                for server_name, server_config in local_servers.items():
                    # Check if this server already exists (from hub)
                    if server_name in self.server_configs:
                        overridden_servers.append(server_name)
                        logger.info(
                            f'🔄 Local config overrides hub server: {server_name}'
                        )

                    # Local config takes precedence in merge mode
                    self.server_configs[server_name] = server_config

                local_servers_count = len(local_servers)
                logger.info(
                    f'📁 Loaded {local_servers_count} servers from local config'
                )

                if overridden_servers:
                    logger.info(
                        f"🔄 Overridden hub servers: {', '.join(overridden_servers)}"
                    )

            except Exception as e:
                logger.warning(f'❌ Failed to load local config: {e}')

        # Ensure at least one source succeeded
        if not self.server_configs:
            raise ValueError(
                'No server configurations could be loaded from any source')

        logger.info(
            f'🔀 Merge complete: Hub({hub_servers_count}) + Local({local_servers_count}) '
            f'= Total({len(self.server_configs)}) servers')

    async def _load_configs_remote(self) -> None:
        """Mode 3: Only load from ModelScope Hub, ignore local config"""

        hub_servers_count = 0

        # Only load from ModelScope Hub
        try:
            hub_configs = self.api.list_operational_mcp_servers(
                token=self.token, endpoint=None)

            mcp_servers = hub_configs.get('mcpServers', {})
            for server_name, server_config in mcp_servers.items():
                self.server_configs[server_name] = server_config

            hub_servers_count = len(mcp_servers)
            logger.info(
                f'🌐 Loaded {hub_servers_count} servers from ModelScope Hub (remote-only mode)'
            )

        except Exception as e:
            logger.error(f'❌ Failed to load from ModelScope Hub: {e}')

        # Warn if local config is provided but ignored
        if self.config:
            logger.warning(
                '⚠️ Local config provided but ignored in remote-only mode')

        # Ensure at least one source succeeded
        if not self.server_configs:
            raise ValueError(
                'No server configurations could be loaded from ModelScope Hub')

        logger.info(
            f'🌐 Remote-only mode complete: {hub_servers_count} servers loaded')

    def _deduplicate_by_url(self) -> None:
        """Remove duplicate servers based on URL to avoid connecting to same endpoint twice"""
        url_to_server = {}  # url -> server_name
        duplicates_removed = []

        # Build URL mapping and detect duplicates
        servers_to_remove = []
        for server_name, config in self.server_configs.items():
            url = config.get('url')
            if url:
                if url in url_to_server:
                    # Duplicate URL found
                    existing_server = url_to_server[url]
                    duplicates_removed.append({
                        'removed': server_name,
                        'kept': existing_server,
                        'url': url
                    })
                    servers_to_remove.append(server_name)
                    logger.warning(
                        f"Removing duplicate server '{server_name}' "
                        f"(same URL as '{existing_server}'): {url}")
                else:
                    url_to_server[url] = server_name

        # Remove duplicates
        for server_name in servers_to_remove:
            del self.server_configs[server_name]

        if duplicates_removed:
            logger.info(
                f'Removed {len(duplicates_removed)} duplicate servers based on URL'
            )

    def _parse_local_config(
            self, config: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Parse local configuration from file or dict"""
        if isinstance(config, dict):
            return config
        elif isinstance(config, str):
            with open(config, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise ValueError('Config must be a file path string or dictionary')

    async def _discover_tools(self) -> None:
        """Discover tools from all configured servers using ephemeral connections"""

        # Use ThreadPoolExecutor for concurrent connections
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit connection tasks
            future_to_server = {
                executor.submit(self._discover_server_tools, server_name,
                                server_config): server_name
                for server_name, server_config in self.server_configs.items()
            }

            # Collect results
            for future in as_completed(future_to_server):
                server_name = future_to_server[future]
                try:
                    server_tools = future.result()
                    if server_tools:
                        # Add server prefix to tool names in the tool list
                        prefixed_tools = []
                        for tool in server_tools:
                            original_name = tool.get('name')
                            if original_name:
                                prefixed_tool = tool.copy()
                                prefixed_tool[
                                    'name'] = f'{server_name}:{original_name}'
                                prefixed_tool['original_name'] = original_name
                                prefixed_tools.append(prefixed_tool)

                        self.tool_list.append({
                            'server_name': server_name,
                            'tools': prefixed_tools
                        })

                        # Update tool-to-server mapping with server prefix to avoid conflicts
                        for tool in server_tools:
                            original_tool_name = tool.get('name')
                            if original_tool_name:
                                # Use "servername:toolname" format to avoid conflicts
                                prefixed_tool_name = f'{server_name}:{original_tool_name}'
                                self.tool_to_server[prefixed_tool_name] = {
                                    'server_name': server_name,
                                    'original_tool_name': original_tool_name
                                }

                        self.stats['successful_connections'] += 1
                        self.stats['total_tools'] += len(server_tools)

                        logger.info(
                            f'Discovered {len(server_tools)} tools from {server_name}'
                        )
                    else:
                        self.stats['failed_connections'] += 1
                        logger.warning(
                            f'No tools discovered from {server_name}')

                except Exception as e:
                    self.stats['failed_connections'] += 1
                    logger.error(
                        f'Failed to discover tools from {server_name}: {e}')

    def _discover_server_tools(
            self, server_name: str,
            server_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Discover tools from a single server (synchronous for thread execution)"""
        try:
            # Create temporary client for tool discovery (always use ephemeral during warmup)
            client = MCPClient(server_config)

            # Use asyncio.run to handle async operations in thread
            async def get_tools():
                async with client:
                    tools = await client.list_tools()
                    return [{
                        'name':
                        tool.name,
                        'description':
                        tool.description[:100] if tool.description else '',
                        'full_description':
                        tool.description or '',
                        'input_schema':
                        getattr(tool, 'inputSchema', None)
                    } for tool in tools]

            return asyncio.run(get_tools())

        except Exception as e:
            logger.error(f'Error discovering tools from {server_name}: {e}')
            return []

    def _generate_tools_prompt(self) -> None:
        """Generate a prompt string for LLM consumption"""
        if not self.tool_list:
            self.tool_prompt = 'No tools available.'
            return

        prompt_lines = [f"Total available tools: {self.stats['total_tools']}"]

        for server_info in self.tool_list:
            server_name = server_info['server_name']
            tools = server_info['tools']
            strategy = self._connection_strategies.get(server_name,
                                                       'ephemeral')

            prompt_lines.append(
                f'\n{server_name} ({strategy}): {len(tools)} tools')

            for tool in tools:
                # Use first 50 characters of description
                desc = tool['description'][:50] + '...' if len(
                    tool['description']) > 50 else tool['description']
                prompt_lines.append(f"  - {tool['name']}: {desc}")

        self.tool_prompt = '\n'.join(prompt_lines)
        logger.debug(
            f'Generated tools prompt ({len(self.tool_prompt)} characters)')

    async def call_tools(
            self,
            function_calls: Union[Dict[str, Any], List[Dict[str, Any]]],
            framework: Optional[str] = 'auto'
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Unified interface for calling one or multiple tools with intelligent connection management

        This method provides a single entry point for executing tools on any connected MCP server.
        It supports multiple LLM framework formats, concurrent execution, and automatic error handling.
        Connection strategies are optimized per server type for best performance.

        Args:
            function_calls (Union[Dict, List[Dict]]): Tool call(s) to execute
                Single call format: {"name": "server:tool", "arguments": {...}}
                Multiple calls format: [{"name": "server1:tool1", ...}, {"name": "server2:tool2", ...}]
            framework (str, optional): LLM framework format detection (default: "auto")
                - "auto": Automatically detect format from call structure
                - "openai": OpenAI function calling format
                - "langchain": LangChain tool format
                - "transformers": HuggingFace Transformers format
                - "default": Simple name/arguments format

        Returns:
            Union[str, List[Dict]]: Execution results
                Single call: Tool result as string
                Multiple calls: List of result dictionaries with success/error status

        Raises:
            ValueError: If tool name is invalid or server not found
            ConnectionError: If unable to connect to target server
            TimeoutError: If tool execution exceeds timeout

        Single Tool Call Examples:
            >>> # Basic tool call (simple format)
            >>> result = await mcp.call_tools({
            ...     "name": "calculator:add",
            ...     "arguments": {"a": 5, "b": 3}
            ... })
            >>> print(result)  # "8"

            >>> # OpenAI function calling format
            >>> openai_call = {
            ...     "name": "weather:get_current",
            ...     "arguments": '{"location": "Beijing", "unit": "celsius"}'
            ... }
            >>> result = await mcp.call_tools(openai_call)
            >>> print(result)  # "Temperature in Beijing: 22°C, Sunny"


            >>> # Transformers function calling format
            >>> transformers_call = {
            ...     "function": {
            ...         "name": "translator:translate",
            ...         "parameters": {"text": "Hello", "target_lang": "zh"}
            ...     }
            ... }
            >>> result = await mcp.call_tools(transformers_call)

        Multiple Tool Calls (Concurrent Execution):
            >>> # Execute multiple tools concurrently
            >>> calls = [
            ...     {"name": "calculator:add", "arguments": {"a": 10, "b": 5}},
            ...     {"name": "weather:get_current", "arguments": {"location": "Shanghai"}},
            ...     {"name": "translator:translate", "arguments": {"text": "Good morning", "target": "fr"}}
            ... ]
            >>> results = await mcp.call_tools(calls)
            >>>
            >>> # Process results
            >>> for result in results:
            ...     if result['success']:
            ...         print(f"{result['tool_name']}: {result['result']}")
            ...     else:
            ...         print(f"Error in {result['tool_name']}: {result['error']}")

        Framework-Specific Examples:
            >>> # OpenAI GPT function calls
            >>> openai_calls = [
            ...     {
            ...         "type": "function",
            ...         "function": {
            ...             "name": "search:web",
            ...             "arguments": '{"query": "latest AI news"}'
            ...         }
            ...     }
            ... ]
            >>> results = await mcp.call_tools(openai_calls)

            >>> # Claude tool use format
            >>> claude_call = {
            ...     "tool_use": {
            ...         "name": "calculator:multiply",
            ...         "parameters": {"x": 7, "y": 6}
            ...     }
            ... }
            >>> result = await mcp.call_tools(claude_call)

        Error Handling Patterns:
            >>> # Handle specific errors
            >>> try:
            ...     result = await mcp.call_tools({
            ...         "name": "nonexistent:tool",
            ...         "arguments": {}
            ...     })
            ... except ValueError as e:
            ...     print(f"Tool not found: {e}")
            ... except TimeoutError as e:
            ...     print(f"Tool execution timed out: {e}")
            ... except Exception as e:
            ...     print(f"Unexpected error: {e}")

            >>> # Batch error handling
            >>> results = await mcp.call_tools([
            ...     {"name": "valid:tool", "arguments": {}},
            ...     {"name": "invalid:tool", "arguments": {}}
            ... ])
            >>> successful = [r for r in results if r['success']]
            >>> failed = [r for r in results if not r['success']]

        Performance Considerations:
            >>> # Connection strategies are automatically optimized:
            >>> # - STDIO servers: New process per call (ephemeral)
            >>> # - SSE servers: Reuse persistent connections
            >>> # - HTTP servers: Ephemeral connections (future optimization)
            >>>
            >>> # Check active strategies:
            >>> stats = mcp.get_stats()
            >>> print(f"Persistent connections: {stats['active_persistent_connections']}")
            >>> print(f"Connection strategies: {stats['connection_strategies']}")

        Tool Naming Rules:
            >>> # Tools are prefixed with server names to avoid conflicts
            >>> # Format: "servername:toolname"
            >>> #
            >>> # Original: server "google" has tool "search"
            >>> # Prefixed: "google:search"
            >>> #
            >>> # Use mcp.list_tools() to see all available prefixed names
            >>> print(mcp.list_tools())

        Advanced Usage:
            >>> # Tool calls with complex arguments
            >>> complex_call = {
            ...     "name": "database:query",
            ...     "arguments": {
            ...         "sql": "SELECT * FROM users WHERE age > ?",
            ...         "params": [18],
            ...         "limit": 100
            ...     }
            ... }
            >>> result = await mcp.call_tools(complex_call)

            >>> # Mixed successful and failed calls
            >>> mixed_calls = [
            ...     {"name": "working:tool", "arguments": {}},
            ...     {"name": "broken:tool", "arguments": {}},
            ...     {"name": "another:working", "arguments": {}}
            ... ]
            >>> results = await mcp.call_tools(mixed_calls)
            >>> success_count = sum(1 for r in results if r['success'])
            >>> print(f"Success rate: {success_count}/{len(results)}")

        Note:
            - All tool execution is asynchronous and non-blocking
            - Multiple calls are executed concurrently for better performance
            - Connection failures trigger automatic retry with fresh connections
            - Tool arguments can be dict, JSON string, or complex nested objects
            - Results always include execution metadata for debugging
        """

        # Handle single function call
        if isinstance(function_calls, dict):
            return await self._execute_single_tool(function_calls, framework)

        # Handle multiple function calls
        elif isinstance(function_calls, list):
            return await self._execute_multiple_tools(function_calls,
                                                      framework)

        else:
            raise ValueError(
                'function_calls must be a dict (single call) or list (multiple calls)'
            )

    async def _execute_single_tool(self, function_call: Dict[str, Any],
                                   framework: str) -> str:
        """Execute a single tool call with connection strategy optimization"""
        try:
            # Normalize function call to standard format
            normalized = MCPRequestAdapter.normalize_function_call(
                function_call)
            tool_name = normalized['tool_name']
            tool_args = normalized['tool_args']

            # Find server for this tool (tool_name should be "servername:toolname")
            tool_info = self.tool_to_server.get(tool_name)
            if not tool_info:
                raise ValueError(f'No server found for tool: {tool_name}')

            server_name = tool_info['server_name']
            original_tool_name = tool_info['original_tool_name']

            # Execute tool using optimized connection strategy
            result = await self._execute_tool_with_strategy(
                server_name, original_tool_name, tool_args)

            self.stats['tool_calls'] += 1
            return result

        except Exception as e:
            logger.error(f'Tool execution failed: {e}')
            return f'Error executing tool: {e}'

    async def _execute_multiple_tools(self, function_calls: List[Dict[str,
                                                                      Any]],
                                      framework: str) -> List[Dict[str, Any]]:
        """Execute multiple tools concurrently (all using same framework)"""
        tasks = []

        for i, function_call in enumerate(function_calls):
            # Each item in the list is a direct function call
            call_id = f'call_{i}'
            task = self._execute_single_tool_async(function_call, framework,
                                                   call_id)
            tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    async def _execute_single_tool_async(
            self,
            function_call: Dict[str, Any],
            framework: str,
            call_id: str = 'unknown') -> Dict[str, Any]:
        """Execute a single tool asynchronously with error handling"""
        try:
            # Normalize function call to standard format
            normalized = MCPRequestAdapter.normalize_function_call(
                function_call)
            tool_name = normalized['tool_name']
            tool_args = normalized['tool_args']

            # Find server for this tool
            tool_info = self.tool_to_server.get(tool_name)
            if not tool_info:
                raise ValueError(f'No server found for tool: {tool_name}')

            server_name = tool_info['server_name']
            original_tool_name = tool_info['original_tool_name']

            # Execute tool with strategy optimization
            result = await self._execute_tool_with_strategy(
                server_name, original_tool_name, tool_args)

            self.stats['tool_calls'] += 1
            return {
                'call_id': call_id,
                'tool_name': tool_name,
                'success': True,
                'result': result
            }

        except Exception as e:
            logger.error(f'Async tool execution failed for {call_id}: {e}')
            return {
                'call_id': call_id,
                'tool_name':
                tool_name if 'tool_name' in locals() else 'unknown',
                'success': False,
                'error': str(e)
            }

    async def _execute_tool_with_strategy(self, server_name: str,
                                          original_tool_name: str,
                                          tool_args: Dict[str, Any]) -> Any:
        """Execute tool using the optimal connection strategy for the server"""

        # Get server configuration and strategy
        server_config = self.server_configs.get(server_name)
        if not server_config:
            raise ValueError(f'Server configuration not found: {server_name}')

        strategy = self._connection_strategies.get(server_name, 'ephemeral')

        if strategy == 'persistent':
            # Use persistent connection strategy
            return await self._execute_tool_persistent(server_name,
                                                       server_config,
                                                       original_tool_name,
                                                       tool_args)
        else:
            # Use ephemeral connection strategy (default)
            return await self._execute_tool_ephemeral(server_config,
                                                      original_tool_name,
                                                      tool_args)

    async def _execute_tool_ephemeral(self, server_config: Dict[str, Any],
                                      original_tool_name: str,
                                      tool_args: Dict[str, Any]) -> Any:
        """Execute tool using ephemeral connection (create-connect-use-disconnect)"""

        # Create new client for this single operation
        client = MCPClient(server_config)

        try:
            async with client:
                result = await client.call_tool(original_tool_name, tool_args)
                self.stats['ephemeral_connections'] += 1
                logger.debug(
                    f'Tool {original_tool_name} executed using ephemeral connection'
                )
                return result

        except Exception as e:
            logger.error(
                f'Failed to execute {original_tool_name} with ephemeral connection: {e}'
            )
            raise

    async def _execute_tool_persistent(self, server_name: str,
                                       server_config: Dict[str, Any],
                                       original_tool_name: str,
                                       tool_args: Dict[str, Any]) -> Any:
        """Execute tool using persistent connection (reuse existing connection if available)"""

        # Check if we have an existing persistent client
        if server_name not in self.clients:
            # Create new persistent client
            client = MCPClient(server_config)
            await client.connect()
            self.clients[server_name] = client
            self.stats['persistent_connections'] += 1
            logger.debug(
                f'Created new persistent connection for {server_name}')

        client = self.clients[server_name]

        # Update usage tracking
        self._client_usage_count[server_name] = self._client_usage_count.get(
            server_name, 0) + 1
        self._last_used_time[server_name] = time.time()

        try:
            # Check if connection is still active
            if not client.is_connected():
                logger.warning(
                    f'Persistent connection to {server_name} was lost, reconnecting...'
                )
                await client.connect()

            result = await client.call_tool(original_tool_name, tool_args)
            logger.debug(
                f'Tool {original_tool_name} executed using persistent connection to {server_name}'
            )
            return result

        except Exception as e:
            logger.error(
                f'Failed to execute {original_tool_name} on persistent connection '
                f'to {server_name}: {e}')

            # Try to recover by recreating the connection
            try:
                logger.info(
                    f'Attempting to recover persistent connection to {server_name}'
                )
                await client.disconnect()
                await client.connect()
                self.clients[server_name] = client

                # Retry the tool call
                result = await client.call_tool(original_tool_name, tool_args)
                logger.info(
                    f'Successfully recovered and executed {original_tool_name} on {server_name}'
                )
                return result

            except Exception as recovery_error:
                logger.error(
                    f'Failed to recover persistent connection to {server_name}: {recovery_error}'
                )
                # Remove the failed client and let it be recreated next time
                if server_name in self.clients:
                    del self.clients[server_name]
                raise

    def get_tools_prompt(self) -> str:
        """Get the generated tools prompt for LLM"""
        return self.tool_prompt

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of all available tools with their connection strategies"""
        all_tools = []
        for server_info in self.tool_list:
            server_name = server_info['server_name']
            strategy = self._connection_strategies.get(server_name,
                                                       'ephemeral')
            for tool in server_info['tools']:
                tool_copy = tool.copy()
                tool_copy['server'] = server_name
                tool_copy['connection_strategy'] = strategy
                all_tools.append(tool_copy)
        return all_tools

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics including connection strategy information"""
        base_stats = self.stats.copy()

        # Add connection strategy statistics
        strategy_stats = {}
        for server_name, strategy in self._connection_strategies.items():
            if strategy not in strategy_stats:
                strategy_stats[strategy] = 0
            strategy_stats[strategy] += 1

        base_stats['connection_strategies'] = strategy_stats
        base_stats['active_persistent_connections'] = len(self.clients)
        base_stats['server_usage_counts'] = self._client_usage_count.copy()

        return base_stats

    def get_duplicate_info(self) -> Dict[str, Any]:
        """Get information about detected duplicates and conflicts"""
        # Tool conflicts
        tool_conflicts = []
        tool_counts = {}

        # Count tools per server
        for server_info in self.tool_list:
            server_name = server_info['server_name']
            tools = server_info['tools']
            tool_counts[server_name] = len(tools)

            # Check for tool name conflicts across servers
            for tool in tools:
                tool_name = tool['name']
                assigned_server = self.tool_to_server.get(tool_name)
                if assigned_server != server_name:
                    tool_conflicts.append({
                        'tool_name': tool_name,
                        'assigned_to': assigned_server,
                        'also_in': server_name
                    })

        return {
            'tool_conflicts': tool_conflicts,
            'tool_counts_per_server': tool_counts,
            'total_unique_tools': len(self.tool_to_server),
            'servers_after_dedup': len(self.server_configs)
        }

    def list_tools(self) -> str:
        """
        List all available tools with server information, connection strategies, and usage statistics

        This method provides a comprehensive overview of all discovered tools across all connected
        servers. It includes server details, connection strategies, usage counts, and practical
        examples for calling each tool.

        Returns:
            str: Formatted tool list with complete information for developers
                - Server grouping with connection strategy
                - Tool names (prefixed format)
                - Tool descriptions
                - Usage examples for each tool
                - Usage statistics where available

        Examples:
            >>> # Basic tool listing
            >>> print(mcp.list_tools())
            # Output:
            # 📋 Available Tools List (15 tools)
            # ==================================================
            #
            # 🔹 calculator (3 tools, ephemeral strategy)
            #   • calculator:add
            #     Add two numbers together
            #     Usage: mcp.call_tools({'name': 'calculator:add', 'arguments': {}})
            #
            #   • calculator:subtract
            #     Subtract two numbers
            #     Usage: mcp.call_tools({'name': 'calculator:subtract', 'arguments': {}})


            >>> # LLM prompt preparation
            >>> system_prompt = f'''
            ... You have access to the following tools:
            ... {mcp.list_tools()}
            ...
            ... Use these tools to help users with their requests.
            ... '''

            >>> # Tool discovery for applications
            >>> tools_info = mcp.list_tools()
            >>> available_servers = []
            >>> for line in tools_info.split('\n'):
            ...     if '🔹' in line and 'tools,' in line:
            ...         server_name = line.split('🔹')[1].split('(')[0].strip()
            ...         available_servers.append(server_name)
            >>> print(f"Available servers: {available_servers}")

        Information Included:
            - Total tool count across all servers
            - Server names with tool counts
            - Connection strategy per server (ephemeral/persistent)
            - Usage statistics for frequently used servers
            - Individual tool names in callable format (server:tool)
            - Tool descriptions for understanding functionality
            - Copy-paste ready usage examples

        Output Format:
            📋 Available Tools List (X tools)
            ==================================================

            🔹 server_name (N tools, strategy_type strategy)
                Usage count: X (if > 0)
              • server_name:tool_name
                Tool description text
                Usage: mcp.call_tools({'name': 'server_name:tool_name', 'arguments': {}})

        Performance Notes:
            >>> # This method is fast as it uses cached tool information
            >>> # No network calls are made - data comes from initialization
            >>> start_time = time.time()
            >>> tools = mcp.list_tools()
            >>> elapsed = time.time() - start_time
            >>> print(f"Retrieved {len(tools)} chars in {elapsed:.3f}s")

        Integration Examples:
            >>> # Web API endpoint
            >>> @app.route('/tools')
            >>> def get_tools():
            ...     return {'tools': mcp.list_tools()}

            >>> # CLI interface
            >>> if args.list_tools:
            ...     print(mcp.list_tools())
            ...     sys.exit(0)

            >>> # Interactive exploration
            >>> tools = mcp.list_tools()
            >>> # Search for specific functionality
            >>> if "translate" in tools.lower():
            ...     print("Translation tools are available")
            >>> if "weather" in tools.lower():
            ...     print("Weather tools are available")

        Note:
            - Tool information is cached from initialization - always current
            - Connection strategies shown reflect actual optimization settings
            - Usage counts track calls made since initialization
            - Tool names are guaranteed to be unique and callable
            - Descriptions come directly from the tool servers
        """

        lines = []
        lines.append(
            f"📋 Available Tools List ({self.stats['total_tools']} tools)")
        lines.append('=' * 50)

        for server_info in self.tool_list:
            server_name = server_info['server_name']
            strategy = self._connection_strategies.get(server_name,
                                                       'ephemeral')
            usage_count = self._client_usage_count.get(server_name, 0)

            lines.append(
                f"\n🔹 {server_name} ({len(server_info['tools'])} tools, {strategy} strategy)"
            )
            if usage_count > 0:
                lines.append(f'    Usage count: {usage_count}')

            for tool in server_info['tools']:
                tool_name = tool['name']
                description = tool.get('description', 'No description')

                lines.append(f'  • {tool_name}')
                lines.append(f'    {description}')
                lines.append(
                    f"    Usage: mcp.call_tools({{'name': '{tool_name}', 'arguments': '{{}}'}}))"
                )
                lines.append('')

        return '\n'.join(lines)

    def shutdown(self) -> None:
        """Cleanup resources including persistent connections"""
        # Close persistent clients
        for server_name, client in self.clients.items():
            try:
                # Check if we're in an event loop
                try:
                    loop = asyncio.get_running_loop()
                    # If we're in a running loop, schedule the disconnect
                    loop.create_task(client.disconnect())
                    # Don't wait for completion to avoid blocking
                    logger.debug(
                        f'Scheduled disconnect for persistent connection to {server_name}'
                    )
                except RuntimeError:
                    # No running loop, safe to use asyncio.run()
                    asyncio.run(client.disconnect())
                    logger.debug(
                        f'Closed persistent connection to {server_name}')
            except Exception as e:
                logger.warning(
                    f'Error closing persistent client for {server_name}: {e}')

        self.clients.clear()
        self._client_usage_count.clear()
        self._last_used_time.clear()

        logger.info('MCP Manager shutdown completed - all connections closed')


# Export the main class
__all__ = ['MCP']
