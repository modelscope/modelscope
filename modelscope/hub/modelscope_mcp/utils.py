# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Utility functions for MCP (Model Context Protocol) integration.

This module provides utility functions for working with MCP configurations,
JSON processing, and tool name formatting.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import json

# Configure logging
logger = logging.getLogger(__name__)


class MCPUtilsError(Exception):
    """Base exception for MCP utility errors."""
    pass


class MCPConfigError(MCPUtilsError):
    """Exception raised when MCP configuration is invalid."""
    pass


class MCPJSONError(MCPUtilsError):
    """Exception raised when JSON processing fails."""
    pass


def fix_json_brackets(json_str: str) -> str:
    """
    Fix malformed JSON string by correcting bracket matching.

    This function processes a JSON string that may have mismatched brackets
    and attempts to fix them by adding missing closing brackets or removing
    unmatched opening brackets.

    Args:
        json_str: Input JSON string that may have bracket issues

    Returns:
        Fixed JSON string with proper bracket matching

    Raises:
        MCPJSONError: If the input is not a string

    Example:
        >>> fix_json_brackets('{"key": "value"')
        '{"key": "value"}'
        >>> fix_json_brackets('{"key": "value"}}')
        '{"key": "value"}'
    """
    if not isinstance(json_str, str):
        raise MCPJSONError('Input must be a string')

    # Initialize stack and result string
    stack: List[str] = []
    result: List[str] = []

    # Process each character in the string
    for char in json_str:
        if char in '{[':
            # If it's an opening bracket, push to stack and add to result
            stack.append(char)
            result.append(char)
        elif char in '}]':
            # If it's a closing bracket
            if not stack:
                # If stack is empty, skip this closing bracket
                continue

            # Check if brackets match
            if (char == '}' and stack[-1] == '{') or (char == ']'
                                                      and stack[-1] == '['):
                # Brackets match, pop from stack and add to result
                stack.pop()
                result.append(char)
            else:
                # Brackets don't match, skip this closing bracket
                continue
        else:
            # Other characters are added directly to result
            result.append(char)

    # Handle remaining opening brackets in stack
    while stack:
        # Add corresponding closing bracket for each unmatched opening bracket
        open_bracket = stack.pop()
        result.append('}' if open_bracket == '{' else ']')

    return ''.join(result)


def validate_mcp_config(config: Dict[str, Any]) -> bool:
    """
    Validate MCP configuration structure.

    This function validates that the provided configuration dictionary
    has the correct structure for MCP servers.
    Supports both local process and remote service configurations.

    Args:
        config: MCP configuration dictionary

    Returns:
        True if configuration is valid, False otherwise

    Raises:
        MCPConfigError: If the configuration is invalid

    Example:
        >>> # Local process service
        >>> config = {
        ...     'filesystem': {
        ...         'command': 'npx',
        ...         'args': ['-y', '@modelcontextprotocol/server-filesystem']
        ...     }
        ... }
        >>> validate_mcp_config(config)
        True

        >>> # Remote service
        >>> config = {
        ...     'amap-maps': {
        ...         'type': 'sse',
        ...         'url': 'https://example.com/sse'
        ...     }
        ... }
        >>> validate_mcp_config(config)
        True
    """
    if not isinstance(config, dict):
        raise MCPConfigError('Configuration must be a dictionary')

    # 支持两种配置格式：
    # 1. 直接的服务配置: {"server_name": {...}}
    # 2. 嵌套的 mcpServers 配置: {"mcpServers": {"server_name": {...}}}

    if 'mcpServers' in config:
        # 嵌套格式
        mcp_servers = config['mcpServers']
        if not isinstance(mcp_servers, dict):
            raise MCPConfigError("'mcpServers' must be a dictionary")
    else:
        # 直接格式
        mcp_servers = config

    for server_name, server_config in mcp_servers.items():
        if not isinstance(server_name, str):
            raise MCPConfigError(
                f"Server name '{server_name}' must be a string")

        if not isinstance(server_config, dict):
            raise MCPConfigError(
                f"Server configuration for '{server_name}' must be a dictionary"
            )

        # Check for required fields based on connection type
        if 'command' in server_config:
            # stdio connection
            if not isinstance(server_config['command'], str):
                raise MCPConfigError(
                    f"Command for server '{server_name}' must be a string")
            if 'args' not in server_config or not isinstance(
                    server_config['args'], list):
                raise MCPConfigError(
                    f"Args for server '{server_name}' must be a list")

            # 可选的环境变量
            if 'env' in server_config and not isinstance(
                    server_config['env'], dict):
                raise MCPConfigError(
                    f"Env for server '{server_name}' must be a dictionary")

        elif 'url' in server_config:
            # sse connection
            if not isinstance(server_config['url'], str):
                raise MCPConfigError(
                    f"URL for server '{server_name}' must be a string")

            # 检查服务类型
            service_type = server_config.get('type', 'http')
            if service_type not in ['sse', 'http', 'streamablehttp']:
                raise MCPConfigError(
                    f"Service type '{service_type}' for server '{server_name}' is not supported. "
                    f'Supported types: sse, http, streamablehttp')
        else:
            # 无效配置
            raise MCPConfigError(
                f"Server '{server_name}' must have either 'command' (for local process) "
                f"or 'url' (for remote service) field")

    return True


def format_mcp_tool_name(server_name: str, tool_name: str) -> str:
    """
    Format MCP tool name with server prefix.

    This function creates a formatted tool name that includes both the
    server name and tool name, separated by '---'.

    Args:
        server_name: Name of the MCP server
        tool_name: Name of the tool

    Returns:
        Formatted tool name

    Raises:
        MCPUtilsError: If server_name or tool_name is empty

    Example:
        >>> format_mcp_tool_name('filesystem', 'listDirectory')
        'filesystem---listDirectory'
    """
    if not server_name:
        raise MCPUtilsError('Server name cannot be empty')
    if not tool_name:
        raise MCPUtilsError('Tool name cannot be empty')

    return f'{server_name}---{tool_name}'


def parse_mcp_tool_name(formatted_name: str) -> Tuple[str, Optional[str]]:
    """
    Parse formatted MCP tool name back to server and tool names.

    This function extracts the server name and tool name from a formatted
    tool name that was created using format_mcp_tool_name().

    Args:
        formatted_name: Formatted tool name

    Returns:
        Tuple of (server_name, tool_name). If no separator is found,
        tool_name will be None.

    Raises:
        MCPUtilsError: If formatted_name is empty

    Example:
        >>> parse_mcp_tool_name('filesystem---listDirectory')
        ('filesystem', 'listDirectory')
        >>> parse_mcp_tool_name('simple_tool')
        ('simple_tool', None)
    """
    if not formatted_name:
        raise MCPUtilsError('Formatted name cannot be empty')

    if '---' in formatted_name:
        parts = formatted_name.split('---', 1)
        return parts[0], parts[1]
    else:
        return formatted_name, None


def validate_tool_arguments(args: Dict[str, Any]) -> bool:
    """
    Validate tool arguments structure.

    This function validates that the provided tool arguments have the
    correct structure and types.

    Args:
        args: Tool arguments dictionary

    Returns:
        True if arguments are valid

    Raises:
        MCPUtilsError: If arguments are invalid
    """
    if not isinstance(args, dict):
        raise MCPUtilsError('Tool arguments must be a dictionary')

    # Check for required fields if this is a schema validation
    if 'type' in args:
        if not isinstance(args['type'], str):
            raise MCPUtilsError("Schema 'type' must be a string")

        if 'properties' in args:
            if not isinstance(args['properties'], dict):
                raise MCPUtilsError("Schema 'properties' must be a dictionary")

        if 'required' in args:
            if not isinstance(args['required'], list):
                raise MCPUtilsError("Schema 'required' must be a list")

    return True


def sanitize_json_string(json_str: str) -> str:
    """
    Sanitize a JSON string by removing invalid characters and fixing common issues.

    This function attempts to clean up a JSON string that may have
    encoding issues or invalid characters.

    Args:
        json_str: Input JSON string

    Returns:
        Sanitized JSON string

    Raises:
        MCPJSONError: If the input is not a string
    """
    if not isinstance(json_str, str):
        raise MCPJSONError('Input must be a string')

    # Remove null bytes and other control characters
    sanitized = ''.join(
        char for char in json_str if ord(char) >= 32 or char in '\n\r\t')

    # Fix common encoding issues
    sanitized = sanitized.replace('\u0000', '')  # Remove null bytes
    sanitized = sanitized.replace('\ufffd',
                                  '')  # Remove replacement characters

    return sanitized


def merge_mcp_configs(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple MCP configurations into a single configuration.

    This function combines multiple MCP configuration dictionaries,
    handling conflicts by using the last configuration for each server.

    Args:
        configs: List of MCP configuration dictionaries

    Returns:
        Merged configuration dictionary

    Raises:
        MCPConfigError: If any configuration is invalid
    """
    if not configs:
        raise MCPConfigError('At least one configuration must be provided')

    merged_config = {'mcpServers': {}}

    for config in configs:
        if not isinstance(config, dict):
            raise MCPConfigError('Each configuration must be a dictionary')

        if 'mcpServers' in config:
            if not isinstance(config['mcpServers'], dict):
                raise MCPConfigError("'mcpServers' must be a dictionary")

            merged_config['mcpServers'].update(config['mcpServers'])

    return merged_config


def extract_server_info(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extract server information from MCP configuration.

    This function extracts server-specific information from the MCP
    configuration and returns it in a structured format.

    Args:
        config: MCP configuration dictionary

    Returns:
        Dictionary mapping server names to their configuration

    Raises:
        MCPConfigError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise MCPConfigError('Configuration must be a dictionary')

    if 'mcpServers' not in config:
        return {}

    mcp_servers = config['mcpServers']
    if not isinstance(mcp_servers, dict):
        raise MCPConfigError("'mcpServers' must be a dictionary")

    server_info = {}
    for server_name, server_config in mcp_servers.items():
        if isinstance(server_config, dict):
            server_info[server_name] = {
                'type': 'stdio' if 'command' in server_config else 'sse',
                'config': server_config
            }

    return server_info


def validate_json_schema(schema: Dict[str, Any]) -> bool:
    """
    Validate JSON schema structure for MCP tools.

    This function validates that a JSON schema has the required fields
    for MCP tool definitions.

    Args:
        schema: JSON schema dictionary

    Returns:
        True if schema is valid

    Raises:
        MCPUtilsError: If schema is invalid
    """
    if not isinstance(schema, dict):
        raise MCPUtilsError('Schema must be a dictionary')

    required_fields = {'type', 'properties'}
    missing_fields = required_fields - set(schema.keys())

    if missing_fields:
        raise MCPUtilsError(
            f'Missing required schema fields: {missing_fields}')

    if not isinstance(schema['type'], str):
        raise MCPUtilsError("Schema 'type' must be a string")

    if not isinstance(schema['properties'], dict):
        raise MCPUtilsError("Schema 'properties' must be a dictionary")

    # Validate required field if present
    if 'required' in schema:
        if not isinstance(schema['required'], list):
            raise MCPUtilsError("Schema 'required' must be a list")

        # Check that all required properties exist in properties
        properties = set(schema['properties'].keys())
        required_props = set(schema['required'])
        missing_props = required_props - properties

        if missing_props:
            raise MCPUtilsError(
                f'Required properties not found in schema: {missing_props}')

    return True
