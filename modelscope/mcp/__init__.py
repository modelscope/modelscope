# Copyright (c) Alibaba, Inc. and its affiliates.
# noqa: isort:skip_file, yapf: disable
from .api import MCPApi

"""
MCP (Model Context Protocol) integration for ModelScope.

This package provides a simple interface to connect to and interact with MCP servers.
The API is designed to be easy to use with minimal configuration required.

Key classes:
- MCPApi: Simple API for listing and managing MCP servers

For advanced MCP protocol features:
- MCPClient: Available via explicit import (from modelscope.hub.mcp.client import MCPClient)
"""

__all__ = [
    'MCPApi'
]
