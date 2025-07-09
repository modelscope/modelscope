# noqa: isort:skip_file, yapf: disable
from modelscope.hub.modelscope_mcp.api import (McpApi, MCPApiError,
                                               MCPApiRequestError,
                                               MCPApiResponseError)
from modelscope.hub.modelscope_mcp.client import (MCPClient, MCPClientError,
                                                  MCPConnectionError,
                                                  MCPResourceError,
                                                  MCPTimeoutError,
                                                  MCPToolExecutionError,
                                                  MCPTransportError)
from modelscope.hub.modelscope_mcp.config_manager import (
    create_default_config, load_config, merge_configs)
from modelscope.hub.modelscope_mcp.manager import (
    MCPManager, MCPManagerConnectionError, MCPManagerError,
    MCPManagerInitializationError, MCPManagerToolExecutionError, MCPTool,
    ServerInfo, ToolInfo, create_mcp_manager)
from modelscope.hub.modelscope_mcp.types import (APIConfig, MCPConfig,
                                                 ServerConfig, ToolArguments,
                                                 ToolName, ToolResult,
                                                 create_client_info,
                                                 format_server_name,
                                                 get_latest_protocol_version,
                                                 is_valid_protocol_version,
                                                 parse_server_name,
                                                 validate_api_config,
                                                 validate_mcp_config,
                                                 validate_tool_arguments)
from modelscope.hub.modelscope_mcp.utils import (MCPConfigError, MCPJSONError,
                                                 MCPUtilsError,
                                                 extract_server_info,
                                                 fix_json_brackets,
                                                 format_mcp_tool_name,
                                                 merge_mcp_configs,
                                                 parse_mcp_tool_name,
                                                 sanitize_json_string,
                                                 validate_json_schema)
from modelscope.hub.modelscope_mcp.utils import \
    validate_tool_arguments as validate_tool_args_utils

"""
MCP (Model Context Protocol) integration for ModelScope.

This package provides integration with the Model Context Protocol,
allowing ModelScope to connect to and interact with MCP servers.
"""

__all__ = [
    # Client classes
    'MCPClient',
    'MCPClientError',
    'MCPConnectionError',
    'MCPToolExecutionError',
    # API classes
    'McpApi',
    'MCPApiError',
    'MCPApiRequestError',
    'MCPApiResponseError',
    # Manager classes
    'MCPManager',
    'MCPManagerError',
    'MCPManagerInitializationError',
    'MCPManagerToolExecutionError',
    'MCPManagerConnectionError',
    'MCPTool',
    'ToolInfo',
    'ServerInfo',
    'create_mcp_manager',
    # MCP Client classes
    'MCPClient',
    'MCPClientError',
    'MCPConnectionError',
    'MCPToolExecutionError',
    'MCPTimeoutError',
    'MCPTransportError',
    'MCPResourceError',
    # Config functions
    'load_config',
    'merge_configs',
    'create_default_config',
    # Type definitions
    'MCPConfig',
    'APIConfig',
    'ServerConfig',
    'ToolName',
    'ToolArguments',
    'ToolResult',
    # Utility functions
    'validate_mcp_config',
    'validate_api_config',
    'validate_tool_arguments',
    'format_server_name',
    'parse_server_name',
    'create_client_info',
    'is_valid_protocol_version',
    'get_latest_protocol_version',
    # Utils functions
    'fix_json_brackets',
    'format_mcp_tool_name',
    'parse_mcp_tool_name',
    'validate_json_schema',
    'sanitize_json_string',
    'merge_mcp_configs',
    'extract_server_info',
    'validate_tool_args_utils',
    # Exception classes
    'MCPUtilsError',
    'MCPConfigError',
    'MCPJSONError',
]

__version__ = '0.1.0'
