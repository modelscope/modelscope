"""
Type definitions for MCP (Model Context Protocol) integration.

This module provides type definitions and constants used throughout
the MCP integration, following the patterns established in the official MCP Python SDK.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from mcp.types import CallToolResult, Implementation, ListToolsResult, Tool

# MCP Client Configuration Types
MCPConfig = Dict[str, Any]
APIConfig = Dict[str, str]
ServerConfig = Dict[str, Any]

# MCP Server Connection Types
ConnectionType = Literal['stdio', 'sse', 'websocket']
ServerStatus = Literal['connected', 'disconnected', 'error']

# MCP Tool Types
ToolName = str
ToolArguments = Dict[str, Any]
ToolResult = str

# MCP Session Types
SessionId = str
ServerName = str

# Default Values
DEFAULT_CLIENT_INFO = Implementation(name='modelscope-mcp', version='0.1.0')
DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 1.0

# Error Codes
ERROR_CODES = {
    'CONNECTION_FAILED': 'MCP_CONNECTION_FAILED',
    'TOOL_EXECUTION_FAILED': 'MCP_TOOL_EXECUTION_FAILED',
    'INVALID_CONFIG': 'MCP_INVALID_CONFIG',
    'SERVER_NOT_FOUND': 'MCP_SERVER_NOT_FOUND',
    'TIMEOUT': 'MCP_TIMEOUT',
}

# Logging Configuration
LOG_LEVELS = {
    'DEBUG': 'DEBUG',
    'INFO': 'INFO',
    'WARNING': 'WARNING',
    'ERROR': 'ERROR',
    'CRITICAL': 'CRITICAL',
}

# MCP Protocol Versions
SUPPORTED_PROTOCOL_VERSIONS = [
    '2025-03-26',  # Latest version
    '2024-12-18',  # Previous version
]

# Configuration Validation
REQUIRED_API_KEYS = ['model', 'api_key', 'model_server']
# Note: REQUIRED_SERVER_KEYS is deprecated, now supports both local process and remote service configurations
# Local process services require "command" field
# Remote services require "url" field

# Tool Execution
MAX_TOOL_EXECUTION_TIME = 300.0  # 5 minutes
MAX_TOOL_ARGUMENTS_SIZE = 1024 * 1024  # 1MB

# Connection Management
MAX_CONCURRENT_CONNECTIONS = 10
CONNECTION_POOL_SIZE = 5

# Response Processing
MAX_RESPONSE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_TOOL_RESULTS = 100

# Type Aliases for Better Readability
MCPClientConfig = Dict[str, Any]
MCPResponse = Union[str, Dict[str, Any], List[Any]]
MCPError = Dict[str, Any]


# Validation Schemas
def validate_mcp_config(config: MCPConfig) -> bool:
    """
    Validate MCP configuration structure.

    Args:
        config: MCP configuration to validate

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValueError('MCP configuration must be a dictionary')

    if not config:
        raise ValueError('MCP configuration cannot be empty')

    # Supports two configuration formats:
    # 1. Direct service configuration: {"server_name": {...}}
    # 2. Nested mcpServers configuration: {"mcpServers": {"server_name": {...}}}

    if 'mcpServers' in config:
        # Nested format
        mcp_servers = config['mcpServers']
        if not isinstance(mcp_servers, dict):
            raise ValueError("'mcpServers' must be a dictionary")
    else:
        # Direct format
        mcp_servers = config

    for server_name, server_config in mcp_servers.items():
        if not isinstance(server_name, str):
            raise ValueError('Server names must be strings')

        if not isinstance(server_config, dict):
            raise ValueError(
                f'Server configuration for {server_name} must be a dictionary')

        # Check service type
        if 'command' in server_config:
            # Local process service (stdio connection)
            if not isinstance(server_config['command'], str):
                raise ValueError(
                    f'Command for server {server_name} must be a string')
        elif 'url' in server_config:
            # Remote service (SSE/HTTP connection)
            if not isinstance(server_config['url'], str):
                raise ValueError(
                    f'URL for server {server_name} must be a string')

            # Check service type
            service_type = server_config.get('type', 'http')
            if service_type not in ['sse', 'http', 'streamablehttp']:
                raise ValueError(
                    f"Unsupported service type '{service_type}' for server {server_name}"
                )
        else:
            # Invalid configuration
            raise ValueError(
                f"Server {server_name} must have either 'command' (for local process) "
                f"or 'url' (for remote service) field")

    return True


def validate_api_config(config: APIConfig) -> bool:
    """
    Validate API configuration structure.

    Args:
        config: API configuration to validate

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValueError('API configuration must be a dictionary')

    missing_keys = [key for key in REQUIRED_API_KEYS if key not in config]
    if missing_keys:
        raise ValueError(
            f'Missing required API configuration keys: {missing_keys}')

    for key in REQUIRED_API_KEYS:
        if not isinstance(config[key], str):
            raise ValueError(f"API configuration key '{key}' must be a string")

    return True


def validate_tool_arguments(args: ToolArguments) -> bool:
    """
    Validate tool arguments.

    Args:
        args: Tool arguments to validate

    Returns:
        True if arguments are valid

    Raises:
        ValueError: If arguments are invalid
    """
    if not isinstance(args, dict):
        raise ValueError('Tool arguments must be a dictionary')

    # Check size limit
    import json
    args_size = len(json.dumps(args).encode('utf-8'))
    if args_size > MAX_TOOL_ARGUMENTS_SIZE:
        raise ValueError(
            f'Tool arguments size ({args_size}) exceeds limit ({MAX_TOOL_ARGUMENTS_SIZE})'
        )

    return True


# Utility Functions
def format_server_name(name: str) -> str:
    """
    Format server name for consistent usage.

    Args:
        name: Raw server name

    Returns:
        Formatted server name
    """
    return name.lower().replace(' ', '-').replace('_', '-')


def parse_server_name(name: str) -> str:
    """
    Parse server name from formatted string.

    Args:
        name: Formatted server name

    Returns:
        Parsed server name
    """
    return name.replace('-', '_').title()


def create_client_info(name: str, version: str) -> Implementation:
    """
    Create client implementation info.

    Args:
        name: Client name
        version: Client version

    Returns:
        Implementation info object
    """
    return Implementation(name=name, version=version)


def is_valid_protocol_version(version: str) -> bool:
    """
    Check if protocol version is supported.

    Args:
        version: Protocol version to check

    Returns:
        True if version is supported
    """
    return version in SUPPORTED_PROTOCOL_VERSIONS


def get_latest_protocol_version() -> str:
    """
    Get the latest supported protocol version.

    Returns:
        Latest protocol version
    """
    return SUPPORTED_PROTOCOL_VERSIONS[0]
