# Copyright (c) Alibaba, Inc. and its affiliates.
# noqa: isort:skip_file, yapf: disable
from .api import McpApi

"""
MCP (Model Context Protocol) integration for ModelScope.

This package provides a simple interface to connect to and interact with MCP servers.
The API is designed to be easy to use with minimal configuration required.

Key classes:
- McpApi: Simple API for listing and managing MCP servers

For advanced MCP protocol features:
- MCPClient: Available via explicit import (from modelscope.hub.mcp.client import MCPClient)
"""

__all__ = [
    'McpApi'
]

__version__ = '0.1.0'
