# Copyright (c) Alibaba, Inc. and its affiliates.
"""
MCP (Model Context Protocol) API interface for ModelScope Hub.

This module provides API interfaces for interacting with MCP servers
through the ModelScope Hub, including server management and deployment.
"""

from typing import Any, Dict, Optional

import requests

from modelscope.hub.api import HubApi, ModelScopeConfig
from modelscope.hub.errors import raise_for_http_status
from modelscope.utils.logger import get_logger

# Configure logging
logger = get_logger()

# MCP API path suffix
MCP_SUFFIX = '/openapi/v1/mcp/servers'


class MCPApiError(Exception):
    """Base exception for MCP API errors."""
    pass


class MCPApiRequestError(MCPApiError):
    """Exception raised when MCP API request fails."""
    pass


class MCPApiResponseError(MCPApiError):
    """Exception raised when MCP API response is invalid."""
    pass


class McpApi(HubApi):
    """
    MCP (Model Context Protocol) API interface class.

    This class provides methods for interacting with MCP servers through
    the ModelScope Hub API, including listing, deploying, and managing servers.

    Usage:

    >>> api = McpApi()
    >>> api.login(access_token="your_token")  # Same as HubApi.login()
    >>> servers = api.list_mcp_servers()  # No token needed
    >>> my_servers = api.list_operational_mcp_servers()  # Token needed


    Note: McpApi inherits login() from HubApi - same functionality, single class convenience.
    Methods have different token requirements - see individual method docs.
    """

    def __init__(self, endpoint: Optional[str] = None) -> None:
        """
        Initialize MCP API.

        Args:
            endpoint: The modelscope server address. Defaults to None (uses default endpoint).
        """
        # Initialize parent HubApi with default settings
        super().__init__(endpoint=endpoint)

        # Create MCP-specific endpoint without modifying the original
        self.mcp_base_url = self.endpoint + MCP_SUFFIX

    def list_mcp_servers(self,
                         token: Optional[str] = None,
                         filters: Optional[Dict[str, Any]] = None,
                         page_number: Optional[int] = 1,
                         page_size: Optional[int] = 20,
                         search: Optional[str] = '') -> Dict[str, Any]:
        """
        List available MCP servers.

        Usage:
        >>> api = McpApi()
        >>> servers = api.list_mcp_servers()  # Public servers
        >>> servers = api.list_mcp_servers(token="your_token")  # With auth

        Authentication:
        - Token: Optional (public servers work without token)
        - Login: Use api.login() once, then no token needed

        Returns:
            {
                'total_counts': 100,
                'servers': [
                    {'name': 'ServerA', 'description': 'This is a demo server for xxx.'},
                    {'name': 'ServerB', 'description': 'This is another demo server.'},
                    ...
                ]
            }
        """
        if page_number < 1:
            raise ValueError('page_number must be greater than 0')
        if page_size < 1:
            raise ValueError('page_size must be greater than 0')

        url = self.mcp_base_url
        headers = self.builder_headers(self.headers)

        # Only add Authorization header if token is provided
        if token:
            headers['Authorization'] = f'Bearer {token}'

        body = {
            'filters': filters or {},
            'page_number': page_number,
            'page_size': page_size,
            'search': search
        }

        try:
            # Get cookies for authentication
            cookies = ModelScopeConfig.get_cookies()
            r = self.session.put(
                url, headers=headers, json=body, cookies=cookies)
            raise_for_http_status(r)
        except requests.exceptions.RequestException as e:
            logger.error(f'Failed to get MCP servers: {e}')
            raise MCPApiRequestError(f'Failed to get MCP servers: {e}') from e

        try:
            resp = r.json()
        except requests.exceptions.JSONDecodeError as e:
            logger.error(f'JSON parsing failed: {e}')
            logger.error(f'Response content: {r.text}')
            raise MCPApiResponseError(f'Invalid JSON response: {e}') from e

        data = resp.get('data', {})
        mcp_server_list = data.get('mcp_server_list', [])
        server_brief_list = [{
            'name': item.get('name', ''),
            'description': item.get('description', '')
        } for item in mcp_server_list]

        return {
            'total_counts': data.get('total_count', 0),
            'servers': server_brief_list
        }

    def list_operational_mcp_servers(self, token: str) -> Dict[str, Any]:
        """
        Get user-hosted MCP server list.

        Usage:
        >>> api = McpApi()
        >>> api.login(access_token="your_token")
        >>> my_servers = api.list_operational_mcp_servers()  # No token needed after login
        >>> # OR without login:
        >>> my_servers = api.list_operational_mcp_servers(token="your_token")

        Authentication:
        - Token: Required (user's private servers)
        - Login: Use api.login() once, then no token needed

        Returns:
            {
                'total_counts': 10,
                'servers': [
                    {'name': 'ServerA', "id": "@Group1/ServerA", 'description': 'This is a demo server for xxx.'},
                    ...
                ],
                'mcpServers': {
                    'serverA': {'type': 'sse', "id": "@Group2/ServerB", 'url': 'https://example.com/serverA/sse'},
                    ...
                }
            }
        """
        url = f'{self.mcp_base_url}/operational'
        headers = self.builder_headers(self.headers)

        # Only add Authorization header if token is provided
        if token:
            headers['Authorization'] = f'Bearer {token}'
        else:
            raise ValueError('token is required')

        try:
            # Get cookies for authentication
            cookies = ModelScopeConfig.get_cookies()
            r = self.session.get(url, headers=headers, cookies=cookies)
            raise_for_http_status(r)
        except requests.exceptions.RequestException as e:
            logger.error(f'Failed to get operational MCP servers: {e}')
            raise MCPApiRequestError(
                f'Failed to get operational MCP servers: {e}') from e

        logger.debug(f'Response status code: {r.status_code}')

        try:
            resp = r.json()
        except requests.exceptions.JSONDecodeError as e:
            logger.error(f'JSON parsing failed: {e}')
            logger.error(f'Response content: {r.text}')
            raise MCPApiResponseError(f'Invalid JSON response: {e}') from e

        data = resp.get('data', {})
        mcp_server_list = data.get('mcp_server_list', [])
        server_brief_list = [{
            'name': item.get('name', ''),
            'id': item.get('id', ''),
            'description': item.get('description', '')
        } for item in mcp_server_list]

        # Convert to MCP configuration format
        mcp_servers = {}
        for server in mcp_server_list:
            server_id = server.get('id', '')
            if server_id.startswith('@'):
                server_name = server_id.split(
                    '/', 1)[1] if '/' in server_id else server_id[1:]
            else:
                server_name = server_id

            operational_urls = server.get('operational_urls', [])
            if server_name and operational_urls:
                mcp_servers[server_name] = {
                    'type':
                    'sse',
                    'url':
                    operational_urls[0]['url'] if isinstance(
                        operational_urls[0], dict) else operational_urls[0]
                }

        return {
            'total_counts': data.get('total_count', 0),
            'servers': server_brief_list,
            'mcpServers': mcp_servers
        }

    def get_mcp_server(self,
                       server_id: str,
                       token: Optional[str] = None) -> Dict[str, Any]:
        """
        Get specific MCP server information.

        Usage:
        >>> api = McpApi()
        >>> server = api.get_mcp_server("@amap/amap-maps")  # Basic info
        >>> server = api.get_mcp_server("@amap/amap-maps", token="your_token")  # Full info

        Authentication:
        - Token: Optional (basic info works without token)
        - Login: Use api.login() once, then no token needed

        Returns:
            {
                'name': 'ServerA',
                'description': 'This is a demo server for xxx.',
                'id': '@demo/serverA',
                'service_config': {
                    'type': 'sse',
                    'url': 'https://example.com/serverA/sse'
                }
            }
        """
        if not server_id:
            raise ValueError('server_id cannot be empty')

        url = f'{self.mcp_base_url}/{server_id}'
        headers = self.builder_headers(self.headers)

        if token:
            headers['Authorization'] = f'Bearer {token}'

        try:
            # Get cookies for authentication
            cookies = ModelScopeConfig.get_cookies()
            r = self.session.get(
                url,
                headers=headers,
                params={'get_operational_url':
                        True},  # Always get operational URLs
                cookies=cookies)
            raise_for_http_status(r)
        except requests.exceptions.RequestException as e:
            logger.error(f'Failed to get MCP server {server_id}: {e}')
            raise MCPApiRequestError(
                f'Failed to get MCP server {server_id}: {e}') from e

        try:
            resp = r.json()
        except requests.exceptions.JSONDecodeError as e:
            logger.error(f'JSON parsing failed: {e}')
            logger.error(f'Response content: {r.text}')
            raise MCPApiResponseError(f'Invalid JSON response: {e}') from e

        data = resp.get('data', {})

        result = {
            'name': data.get('name', ''),
            'description': data.get('description', ''),
            'id': data.get('id', '')
        }

        service_config = {}
        server_id = data.get('id', '')
        if server_id.startswith('@'):
            server_name = server_id.split(
                '/', 1)[1] if '/' in server_id else server_id[1:]
        else:
            server_name = server_id

        operational_urls = data.get('operational_urls', [])
        if server_name and operational_urls:
            service_config = {
                'type':
                'sse',
                'url':
                operational_urls[0]['url'] if isinstance(
                    operational_urls[0], dict) else operational_urls[0]
            }
        result['service_config'] = service_config
        return result
