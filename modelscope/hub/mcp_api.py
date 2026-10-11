# Copyright (c) Alibaba, Inc. and its affiliates.
"""
MCP (Model Context Protocol) API interface for ModelScope Hub.

This module provides a simple interface to interact with
ModelScope MCP plaza (https://www.modelscope.cn/mcp).
"""
from typing import Any, Dict, Optional

from modelscope.hub.api import HubApi
from modelscope.utils.logger import get_logger

# Configure logging
logger = get_logger()

# MCP API path. Retained for backward compatibility with callers that imported
# it; the requests themselves are issued by modelscope_hub, which owns the URL.
MCP_API_PATH = '/openapi/v1/mcp/servers'


def _operational_url_entry(operational_url: Dict[str, Any]) -> Dict[str, str]:
    """Render one hosted endpoint as the {type, url} pair callers expect.

    The transport is the final path segment (``sse`` / ``streamable_http``).
    """
    url = operational_url.get('url') or ''
    return {'type': url.split('/')[-1], 'url': url}


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

    def __init__(self,
                 endpoint: Optional[str] = None,
                 token: Optional[str] = None) -> None:
        """
        Initialize MCP API.

        Args:
            endpoint: The modelscope server address. Defaults to None (uses default endpoint).
            token: Optional access token for Bearer authentication.
        """
        super().__init__(endpoint=endpoint, token=token)

        self.mcp_base_url = self.endpoint + MCP_API_PATH

    def _delegate(self, token: Optional[str] = None):
        """Return the modelscope_hub client that should serve this call.

        The ambient client is reused unless a one-time ``token`` was supplied, in
        which case a dedicated client is built so the override actually applies.
        """
        if token:
            from modelscope_hub.api import HubApi as _HubApi
            return _HubApi(token=token, endpoint=self.endpoint)
        return self._api

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

        try:
            result = self._delegate(token).list_mcp_servers(
                search=search,
                page_number=1,
                page_size=total_count,
                filter=filter,
            )
        except Exception as e:
            logger.error('Failed to get MCP servers: %s', e)
            raise MCPApiRequestError(f'Failed to get MCP servers: {e}') from e

        mcp_config_list = [{
            'name':
            item.get('name') or item.get('Name') or '',
            'id':
            item.get('id') or item.get('Id') or '',
            'description':
            item.get('description') or item.get('Description') or ''
        } for item in result.items]

        return {'total_count': result.total_count, 'servers': mcp_config_list}

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
        try:
            result = self._delegate(token).list_operational_mcp_servers()
        except Exception as e:
            logger.error(f'Failed to get operational MCP servers: {e}')
            raise MCPApiRequestError(
                f'Failed to get operational MCP servers: {e}') from e

        mcp_config_list = []
        for item in result.items:
            mcp_config = {}
            mcp_config['name'] = item.get('name', '')
            mcp_config['id'] = item.get('id', '')
            mcp_config['description'] = item.get('description', '')
            mcp_config['mcp_servers'] = [
                _operational_url_entry(operational_url)
                for operational_url in item.get('operational_urls') or []
            ]
            mcp_config_list.append(mcp_config)
        return {'total_count': result.total_count, 'servers': mcp_config_list}

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
                - servers: Connection configurations, each with type and url

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

        try:
            data = self._delegate(token).get_mcp_server(
                server_id, get_operational_url=True)
        except Exception as e:
            logger.error(f'Failed to get MCP server {server_id}: {e}')
            raise MCPApiRequestError(
                f'Failed to get MCP server {server_id}: {e}') from e

        if not isinstance(data, dict):
            raise MCPApiResponseError(
                f'Invalid response from MCP server {server_id}: {data!r}')

        result = {
            'name': data.get('name', ''),
            'description': data.get('description', ''),
            'id': data.get('id', '')
        }

        server_name = MCPApi._get_server_name_from_id(data.get('id', ''))
        operational_urls = data.get('operational_urls') or []
        mcp_config_list = []
        if server_name and operational_urls:
            mcp_config_list = [
                _operational_url_entry(operational_url)
                for operational_url in operational_urls
            ]

        result['servers'] = mcp_config_list
        return result

    def deploy_mcp_server(self,
                          server_id: str,
                          transport_type: Optional[str] = None,
                          expiration_minutes: Optional[int] = None,
                          auth_check: Optional[bool] = None,
                          env_info: Optional[Dict[str, str]] = None,
                          token: Optional[str] = None) -> Dict[str, Any]:
        """
        Host an MCP server under your own account.

        Args:
            server_id: MCP server ID (e.g., "@amap/amap-maps")
            transport_type: 'sse' (default) or 'streamable_http'
            expiration_minutes: Lifetime in minutes; -1 for no expiration
            auth_check: Require a ModelScope token when connecting to the URL
            env_info: Environment variables, as listed in the server's env_schema
            token: Optional access token for authentication

        Returns:
            Dict describing the deployment::

                {
                    'id': '...',
                    'url': 'https://mcp.api-inference.modelscope.net/{uuid}/sse',
                    'transport_type': 'sse',
                    'auth_required': False,
                    'expiration': '2025-10-01 21:00:00',
                    'accessible': True
                }

        Raises:
            ValueError: If server_id is empty or None
            MCPApiRequestError: If the deployment request fails

        Authentication:
            Required. A token with 'write' permission or higher.
        """
        if not server_id:
            raise ValueError('server_id cannot be empty')

        payload = {
            'transport_type': transport_type,
            'expiration_minutes': expiration_minutes,
            'auth_check': auth_check,
            'env_info': env_info,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            return self._delegate(token).deploy_mcp_server(
                server_id, payload=payload or None)
        except Exception as e:
            logger.error(f'Failed to deploy MCP server {server_id}: {e}')
            raise MCPApiRequestError(
                f'Failed to deploy MCP server {server_id}: {e}') from e

    def undeploy_mcp_server(self,
                            server_id: str,
                            token: Optional[str] = None) -> Dict[str, Any]:
        """
        Tear down an MCP server you are hosting.

        Args:
            server_id: MCP server ID (e.g., "@amap/amap-maps")
            token: Optional access token for authentication

        Raises:
            ValueError: If server_id is empty or None
            MCPApiRequestError: If the request fails

        Authentication:
            Required. A token with 'write' permission or higher.
        """
        if not server_id:
            raise ValueError('server_id cannot be empty')

        try:
            return self._delegate(token).undeploy_mcp_server(server_id)
        except Exception as e:
            logger.error(f'Failed to undeploy MCP server {server_id}: {e}')
            raise MCPApiRequestError(
                f'Failed to undeploy MCP server {server_id}: {e}') from e
