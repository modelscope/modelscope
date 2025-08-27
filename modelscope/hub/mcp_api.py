# Copyright (c) Alibaba, Inc. and its affiliates.
"""
MCP (Model Context Protocol) API interface for ModelScope Hub.

This module provides a simple interface to interact with
ModelScope MCP plaza (https://www.modelscope.cn/mcp).
"""
from typing import Any, Dict, Optional

import requests

from modelscope.hub.api import HubApi
from modelscope.hub.errors import raise_for_http_status
from modelscope.utils.logger import get_logger

# Configure logging
logger = get_logger()

# MCP API path
MCP_API_PATH = '/openapi/v1/mcp/servers'


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

        self.mcp_base_url = self.endpoint + MCP_API_PATH

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
        """Extract server name from server ID."""
        if '/' in server_id:
            return server_id.split('/', 1)[1]
        return server_id

    def list_mcp_servers(self,
                         token: Optional[str] = None,
                         filter: Optional[Dict[str, Any]] = None,
                         total_count: Optional[int] = 20,
                         search: Optional[str] = '') -> Dict[str, Any]:
        """
        List available MCP servers, if (optional) token is presented, this would return private MCP servers as well.

        Args:
            token: Optional access token for authentication
            filter: Optional filters to apply to the search
                - 'category': str, server category, e.g. 'communication'
                - 'tag': str, server tag, e.g. 'social-media'
                - 'is_hosted': bool, server is hosted
                When all three are passed in, the intersection is taken.
            total_count: Number of servers to return, max 100, default 20
            search: Optional search query string,e.g. Chinese service name, English service name, author/owner username
            You can combine `filter` and `search` to retrieve desired MCP servers.

        Returns:
            Dict containing:
                - total_count: Total number of servers
                - servers: List of server dictionaries with name, id, description

        Raises:
            MCPApiRequestError: If API request fails (network, server errors)
            MCPApiResponseError: If response format is invalid or JSON parsing fails

        Authentication:
            Optional, only required if you wish to retrieve private MCP servers.
            You may leverage the token parameter for one-time authentication, or use api.login()

        Returns:
            {
                'total_count': 20,
                'servers': [
                    {'name': 'ServerA', 'id': '@demo/ServerA', 'description': 'This is a demo server for xxx.'},
                    {'name': 'ServerB', 'id': '@demo/ServerB', 'description': 'This is another demo server.'},
                    ...
                ]
            }
        """

        if total_count is None or total_count < 1 or total_count > 100:
            raise ValueError('total_count must be between 1 and 100')

        body = {
            'filter': filter or {},
            'page_number': 1,
            'page_size': total_count,
            'search': search
        }

        try:
            cookies = self.get_cookies(token)
            r = self.session.put(
                url=self.mcp_base_url,
                headers=self.builder_headers(self.headers),
                json=body,
                cookies=cookies)
            raise_for_http_status(r)
        except requests.exceptions.RequestException as e:
            logger.error('Failed to get MCP servers: %s', e)
            raise MCPApiRequestError(f'Failed to get MCP servers: {e}') from e

        data = self._handle_response(r)
        mcp_server_list = data.get('mcp_server_list', [])
        mcp_config_list = [{
            'name': item.get('name', ''),
            'id': item.get('id', ''),
            'description': item.get('description', '')
        } for item in mcp_server_list]

        return {
            'total_count': data.get('total_count', 0),
            'servers': mcp_config_list
        }

    def list_operational_mcp_servers(self,
                                     token: str = None) -> Dict[str, Any]:
        """
        Get list of operational MCP servers that have been triggered hosting service by the user.

        Returns:
            Dict containing:
                - total_counts: Total number of operational servers
                - servers: List of server info with name, id, description

        Raises:
            MCPApiRequestError: If authentication fails or API request fails
            MCPApiResponseError: If response format is invalid or JSON parsing fails

        Returns:
            {
                'total_count': 10,
                'servers': [
                    {
                        'name': 'ServerA',
                        "id": "@Group1/ServerA",
                        'description': 'This is a demo server for xxx.'
                        'mcp_servers': [
                            {
                                'type': 'sse',
                                'url': 'https://mcp.api-inference.modelscope.net/{uuid}/sse'
                            },
                            {
                                'type': 'streamable_http',
                                'url': 'https://mcp.api-inference.modelscope.net/{uuid}/streamable_http'
                            },
                            ...
                        ]
                    },
                    ...
                ]
            }
        """
        url = f'{self.mcp_base_url}/operational'
        headers = self.builder_headers(self.headers)

        try:
            cookies = self.get_cookies(
                access_token=token, cookies_required=True)
            r = self.session.get(url, headers=headers, cookies=cookies)
            raise_for_http_status(r)
        except requests.exceptions.RequestException as e:
            logger.error(f'Failed to get operational MCP servers: {e}')
            raise MCPApiRequestError(
                f'Failed to get operational MCP servers: {e}') from e

        logger.debug(f'Response status code: {r.status_code}')

        data = self._handle_response(r)
        mcp_server_list = data.get('mcp_server_list', [])

        mcp_config_list = []
        for item in mcp_server_list:
            mcp_config = {}
            mcp_config['name'] = item.get('name', '')
            mcp_config['id'] = item.get('id', '')
            mcp_config['description'] = item.get('description', '')
            mcp_config['mcp_servers'] = []
            for operational_url in item.get('operational_urls', []):
                mcp_config['mcp_servers'].append({
                    'type':
                    operational_url.get('url').split('/')[-1],
                    'url':
                    operational_url.get('url', '')
                })
            mcp_config_list.append(mcp_config)
        return {
            'total_count': data.get('total_count', 0),
            'servers': mcp_config_list
        }

    def get_mcp_server(self,
                       server_id: str,
                       token: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed information for a specific MCP Server,
        a valid token shall be provided if the MCP server is private.

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

        Returns:
            {
                'name': 'ServerA',
                'description': 'This is a demo server for xxx.',
                'id': '@demo/serverA',
                'servers': [
                    {
                        'type': 'sse',
                        'url': 'https://mcp.api-inference.modelscope.net/{uuid}/sse'
                    },
                    {
                        'type': 'streamable_http',
                        'url': 'https://mcp.api-inference.modelscope.net/{uuid}/streamable_http'
                    }
                    ...
                ]
            }
        """
        if not server_id:
            raise ValueError('server_id cannot be empty')

        url = f'{self.mcp_base_url}/{server_id}'
        headers = self.builder_headers(self.headers)

        try:
            cookies = self.get_cookies(token)
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

        server_id = data.get('id', '')
        server_name = MCPApi._get_server_name_from_id(server_id)

        operational_urls = data.get('operational_urls', [])
        mcp_config_list = []
        if server_name and operational_urls:
            for operational_url in operational_urls:
                mcp_config = {
                    'type': operational_url.get('url').split('/')[-1],
                    'url': operational_url.get('url', '')
                }
                mcp_config_list.append(mcp_config)

        result['servers'] = mcp_config_list
        return result
