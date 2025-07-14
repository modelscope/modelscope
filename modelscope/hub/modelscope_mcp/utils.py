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


def validate_tool_schema(schema: Dict[str, Any]) -> bool:
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
