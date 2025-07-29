# Copyright (c) Alibaba, Inc. and its affiliates.
# noqa: isort:skip_file, yapf: disable
from .api import McpApi
from .client import MCPClient

"""
MCP (Model Context Protocol) integration for ModelScope.

This package provides a simple interface to connect to and interact with MCP servers.
The API is designed to be easy to use with minimal configuration required.

Key classes:
- McpApi: Simple API for listing and managing MCP servers
- MCPClient: Client for connecting to and using MCP tools
"""

__all__ = [
    'McpApi'
]

__version__ = '0.1.0'
