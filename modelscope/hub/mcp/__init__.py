# Copyright (c) Alibaba, Inc. and its affiliates.
# noqa: isort:skip_file, yapf: disable
from .api import McpApi
from .client import MCPClient

"""
MCP (Model Context Protocol) integration for ModelScope.

This package provides integration with the Model Context Protocol,
allowing ModelScope to connect to and interact with MCP servers.
"""

__all__ = [
    # API classes
    'McpApi',
    'MCPClient'
    # Exception classes
]

__version__ = '0.1.0'
