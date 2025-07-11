# Copyright (c) Alibaba, Inc. and its affiliates.
import requests
from typing import Optional

from modelscope.hub.errors import raise_for_http_status
from modelscope.utils.logger import get_logger

logger = get_logger()

# MCP API path suffix
MCP_API_PATH = '/openapi/v1'


class McpApi:
    """MCP (Model Context Protocol) API interface class"""

    def __init__(self, base_api):
        """
        Initialize MCP API

        Args:
            base_api: HubApi instance for accessing basic API functionality
        """
        self.base_api = base_api
        # Inherit HubApi's endpoint but add MCP-specific path
        self.endpoint = base_api.endpoint + MCP_API_PATH
        self.session = base_api.session
        self.builder_headers = base_api.builder_headers
        self.headers = base_api.headers

    def get_mcp_servers(self,
                        token: str,
                        filter: dict = None,
                        page_number: int = 1,
                        page_size: int = 20,
                        search: str = '',
                        endpoint: Optional[str] = None) -> dict:
        """
        Get MCP server list

        Args:
            token: Authentication token
            filter: Filter condition dictionary containing the following sub-branches:
                - category: String type, filter by category
                - is_hosted: Boolean type, filter by hosting status
                - tag: JSON string type, filter by tags
                When category, is_hosted, and tag are all provided, take the intersection of all three
            page_number: Page number, defaults to 1
            page_size: Page size, defaults to 20
            search: Search keyword, defaults to empty string
            endpoint: API endpoint, defaults to MCP-specific endpoint (inherited from HubApi + /openapi/v1)

        Returns:
            dict: Dictionary containing MCP server list
                - mcp_server_list: Detailed MCP server list
                - total_count: Total count
                - server_brief_list: Brief server information list
        """
        if not endpoint:
            endpoint = self.endpoint
        url = f'{endpoint}/mcp/servers'
        headers = self.builder_headers(self.headers)
        headers['Authorization'] = f'Bearer {token}'

        body = {
            'filter': filter or {},
            'page_number': page_number,
            'page_size': page_size,
            'search': search
        }

        r = self.session.put(url, headers=headers, json=body)
        raise_for_http_status(r)

        try:
            resp = r.json()
        except requests.exceptions.JSONDecodeError:
            logger.error(
                f'Failed to parse JSON response from MCP server list API: {r.text}'
            )
            raise

        data = resp.get('data', {})
        mcp_server_list = data.get('mcp_server_list', [])
        server_brief_list = [{
            'name': item.get('name', ''),
            'description': item.get('description', '')
        } for item in mcp_server_list]
        return {
            'mcp_server_list': mcp_server_list,
            'total_count': data.get('total_count', 0),
            'server_brief_list': server_brief_list
        }

    def get_mcp_server_operational(self,
                                   token: str,
                                   endpoint: Optional[str] = None) -> dict:
        """
        Get user-hosted MCP server list

        Args:
            token: Authentication token
            endpoint: API endpoint, defaults to MCP-specific endpoint (inherited from HubApi + /openapi/v1)

        Returns:
            dict: Dictionary containing MCP server list
                - mcp_server_list: Detailed MCP server list
                - total_count: Total count
                - server_brief_list: Brief server information list
        """
        if not endpoint:
            endpoint = self.endpoint
        url = f'{endpoint}/mcp/servers/operational'
        headers = self.builder_headers(self.headers)
        headers['Authorization'] = f'Bearer {token}'

        r = self.session.get(url, headers=headers)
        raise_for_http_status(r)

        try:
            resp = r.json()
        except requests.exceptions.JSONDecodeError:
            logger.error(
                f'Failed to parse JSON response from MCP server operational API: {r.text}'
            )
            raise

        data = resp.get('data', {})
        mcp_server_list = data.get('mcp_server_list', [])
        server_brief_list = [{
            'name': item.get('name', ''),
            'description': item.get('description', '')
        } for item in mcp_server_list]
        return {
            'mcp_server_list': mcp_server_list,
            'total_count': data.get('total_count', 0),
            'server_brief_list': server_brief_list
        }

    def get_mcp_server_special(self,
                               server_id: str,
                               token: str,
                               get_operational_url: bool = False,
                               endpoint: Optional[str] = None) -> dict:
        """
        Get specific MCP server details

        Args:
            server_id: Server ID
            token: Authentication token
            get_operational_url: Whether to get operational URL, defaults to False
            endpoint: API endpoint, defaults to MCP-specific endpoint (inherited from HubApi + /openapi/v1)

        Returns:
            dict: Dictionary containing MCP server details
        """
        if not endpoint:
            endpoint = self.endpoint
        url = f'{endpoint}/mcp/servers/{server_id}'
        headers = self.builder_headers(self.headers)
        headers['Authorization'] = f'Bearer {token}'
        params = {
            'get_operational_url': str(get_operational_url).lower()
        } if get_operational_url else {}

        r = self.session.get(url, headers=headers, params=params)
        raise_for_http_status(r)

        try:
            resp = r.json()
        except requests.exceptions.JSONDecodeError:
            logger.error(
                f'Failed to parse JSON response from MCP server special API: {r.text}'
            )
            raise
        return resp.get('data', {})
