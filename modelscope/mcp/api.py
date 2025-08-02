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


class MCPApi(HubApi):
    """
    MCP (Model Context Protocol) API interface class.

    This class provides interfaces to interact with ModelScope MCP servers,
    such as to list, deploy and manage MCP servers.

    Note: MCPApi inherits login() from HubApi for authentication.
    Different methods have different token requirements - see individual method docs.
    """

    def __init__(self, endpoint: Optional[str] = None) -> None:
        """
        Initialize MCP API.

        Args:
            endpoint: The modelscope server address. Defaults to None (uses default endpoint).
        """
        super().__init__(endpoint=endpoint)

        self.mcp_base_url = self.endpoint + MCP_SUFFIX

    @staticmethod
    def _handle_response(r: requests.Response) -> Dict[str, Any]:
        """
        Handle HTTP response with unified error handling and JSON parsing.

        Args:
            r: requests Response object

        Returns:
            Parsed response data dict

        Raises:
            MCPApiResponseError: If JSON parsing fails
        """
        try:
            resp = r.json()
        except requests.exceptions.JSONDecodeError as e:
            logger.error(f'JSON parsing failed: {e}')
            logger.error(f'Response content: {r.text}')
            raise MCPApiResponseError(f'Invalid JSON response: {e}') from e

        return resp.get('data', {})

    @staticmethod
    def _get_server_name_from_id(server_id: str) -> str:
        """
        Extract server name from server ID.

        Handles two formats:
        - '@group/server' -> 'server'
        - 'server' -> 'server'

        Args:
            server_id: The server ID to parse

        Returns:
            str: The extracted server name
        """
        if server_id.startswith('@'):
            return server_id.split('/',
                                   1)[1] if '/' in server_id else server_id[1:]
        return server_id

    def list_mcp_servers(self,
                         token: Optional[str] = None,
                         filter: Optional[Dict[str, Any]] = None,
                         page_number: Optional[int] = 1,
                         page_size: Optional[int] = 20,
                         search: Optional[str] = '') -> Dict[str, Any]:
        """
        List available MCP servers, including public and private servers.

        Args:
            token: Optional access token for authentication
            filter: Optional filters to apply to the search
                - 'category': str, server category, e.g. 'communication'
                - 'tag': str, server tag, e.g. 'social-media'
                - 'is_hosted': bool, server is hosted
                When all three are passed in, the intersection is taken.
            page_number: Page number (starts from 1)
            page_size: Number of servers per page
            page_number * page_size <=100
            search: Optional search query string,e.g. Chinese service name, English service name, author/owner username

        Returns:
            Dict containing:
                - total_counts: Total number of servers
                - servers: List of server dictionaries with name, id, description

        Raises:
            ValueError: If page_number < 1 or page_size < 1
            MCPApiRequestError: If API request fails (network, server errors)
            MCPApiResponseError: If response format is invalid or JSON parsing fails

        Authentication:
        - Token: Optional (public servers work without token)
        - Login: Use api.login() once, then no token needed

        Returns:
            {
                'total_counts': 100,
                'servers': [
                    {'name': 'ServerA', 'id': '@demo/ServerA', 'description': 'This is a demo server for xxx.'},
                    {'name': 'ServerB', 'id': '@demo/ServerB', 'description': 'This is another demo server.'},
                    ...
                ]
            }
        """
        if page_number < 1:
            raise ValueError('page_number must be greater than 0')
        if page_size < 1:
            raise ValueError('page_size must be greater than 0')

        # Login if token is provided
        if token:
            self.login(access_token=token)

        body = {
            'filter': filter or {},
            'page_number': page_number,
            'page_size': page_size,
            'search': search
        }

        try:
            cookies = ModelScopeConfig.get_cookies()
            r = self.session.put(
                self.mcp_base_url,
                self.builder_headers(self.headers),
                json=body,
                cookies=cookies)
            raise_for_http_status(r)
        except requests.exceptions.RequestException as e:
            logger.error(f'Failed to get MCP servers: {e}')
            raise MCPApiRequestError(f'Failed to get MCP servers: {e}') from e

        data = self._handle_response(r)
        mcp_server_list = data.get('mcp_server_list', [])
        server_brief_list = [{
            'name': item.get('name', ''),
            'id': item.get('id', ''),
            'description': item.get('description', '')
        } for item in mcp_server_list]

        return {
            'total_counts': data.get('total_count', 0),
            'servers': server_brief_list
        }

    def list_operational_mcp_servers(self,
                                     token: Optional[str] = None
                                     ) -> Dict[str, Any]:
        """
        Get user-hosted MCP server list.

        Args:
            token: Optional access token. If not provided, will use login session cookies.

        Returns:
            Dict containing:
                - total_counts: Total number of operational servers
                - servers: List of server info with name, id, description
                - mcpServers: Dict of server configs ready for MCP client usage

        Raises:
            MCPApiRequestError: If authentication fails or API request fails
            MCPApiResponseError: If response format is invalid or JSON parsing fails

        Authentication:
        - Token: Optional (will try cookies if not provided)
        - Login: Use api.login() once, then no token needed

        Returns:
            {
                'servers': [
                    {
                        'name': 'ServerA',
                        "id": "@Group1/ServerA",
                        'description': 'This is a demo server for xxx.'
                    },
                    ...
                ],
                'mcpServers': {
                    'serverA': {
                        'type': 'sse',
                        "id": "@Group2/ServerB",
                        'url': 'https://mcp.api-inference.modelscope.net//serverA/sse'
                    },
                    ...
                }
            }
        """
        # Login if token is provided
        if token:
            self.login(access_token=token)

        url = f'{self.mcp_base_url}/operational'
        headers = self.builder_headers(self.headers)

        try:
            cookies = ModelScopeConfig.get_cookies()
            r = self.session.get(url, headers=headers, cookies=cookies)
            raise_for_http_status(r)
        except requests.exceptions.RequestException as e:
            # Check if it's an authentication error and provide helpful message
            if '401' in str(e) or 'Unauthorized' in str(e):
                if not token:
                    logger.error(
                        'Authentication failed: No token provided and cookies authentication failed'
                    )
                    raise MCPApiRequestError(
                        'Authentication required: Please provide a token or ensure you are logged in'
                    ) from e
                else:
                    logger.error(
                        f'Authentication failed with provided token: {e}')
                    raise MCPApiRequestError(
                        'Authentication failed: Invalid token or insufficient permissions'
                    ) from e
            else:
                logger.error(f'Failed to get operational MCP servers: {e}')
                raise MCPApiRequestError(
                    f'Failed to get operational MCP servers: {e}') from e

        logger.debug(f'Response status code: {r.status_code}')

        data = self._handle_response(r)
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
            server_name = MCPApi._get_server_name_from_id(server_id)

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
        Get specific MCP server information.Full info including information accessible only with authorization.

        Args:
            server_id: MCP server ID (e.g., "@amap/amap-maps")
            token: Optional access token for authentication

        Returns:
            Dict containing:
                - name: Server name
                - description: Server description
                - id: Server ID
                - service_config: Connection configuration with type and url

        Raises:
            ValueError: If server_id is empty or None
            MCPApiRequestError: If API request fails or server not found
            MCPApiResponseError: If response format is invalid or JSON parsing fails

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
                    'url': 'https://mcp.api-inference.modelscope.net/serverA/sse'
                }
            }
        """
        if not server_id:
            raise ValueError('server_id cannot be empty')

        # Login if token is provided
        if token:
            self.login(access_token=token)

        url = f'{self.mcp_base_url}/{server_id}'
        headers = self.builder_headers(self.headers)

        try:
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

        data = self._handle_response(r)

        result = {
            'name': data.get('name', ''),
            'description': data.get('description', ''),
            'id': data.get('id', '')
        }

        service_config = {}
        server_id = data.get('id', '')
        server_name = MCPApi._get_server_name_from_id(server_id)

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
