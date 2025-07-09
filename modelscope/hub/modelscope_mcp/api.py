# Copyright (c) Alibaba, Inc. and its affiliates.
"""
MCP (Model Context Protocol) API interface for ModelScope Hub.

This module provides API interfaces for interacting with MCP servers
through the ModelScope Hub, including server management and deployment.
"""

import logging
from typing import Any, Dict, List, Optional

import requests

from modelscope.hub.api import HubApi
from modelscope.hub.errors import raise_for_http_status

# Configure logging
logger = logging.getLogger(__name__)

# MCP API path suffix
MCP_API_PATH = '/openapi/v1'


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
    """

    def __init__(self, base_api: HubApi) -> None:
        """
        Initialize MCP API.

        Args:
            base_api: HubApi instance for accessing basic API functionality

        Raises:
            ValueError: If base_api is not a valid HubApi instance
        """

        super().__init__(base_api)
        # Inherit HubApi's endpoint but add OpenAPI-specific path
        self.endpoint = base_api.endpoint + MCP_API_PATH
        self.session = base_api.session
        self.builder_headers = base_api.builder_headers
        self.headers = base_api.headers

    def get_mcp_servers(self,
                        token: Optional[str] = None,
                        filter_dict: Optional[Dict[str, Any]] = None,
                        page_number: int = 1,
                        page_size: int = 20,
                        search: str = '',
                        endpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Get MCP server list.

        Args:
            token: Authentication token (optional). If not provided, request will be made without authentication
            filter_dict: Filter condition dictionary containing the following sub-branches:
                - category: String type, filter by category
                - is_hosted: Boolean type, filter by hosting status
                - tag: JSON string type, filter by tags
                When category, is_hosted, and tag are all provided, take the intersection of all three
            page_number: Page number, defaults to 1
            page_size: Page size, defaults to 20
            search: Search keyword, defaults to empty string
            endpoint: API endpoint, defaults to MCP-specific endpoint (inherited from HubApi + /openapi/v1)

        Returns:
            Dictionary containing MCP server list:
                - mcp_server_list: Detailed MCP server list
                - total_count: Total count
                - server_brief_list: Brief server information list

        Raises:
            MCPApiRequestError: If the API request fails
            MCPApiResponseError: If the API response is invalid
        """
        if page_number < 1:
            raise ValueError('page_number must be greater than 0')
        if page_size < 1:
            raise ValueError('page_size must be greater than 0')

        if not endpoint:
            endpoint = self.endpoint

        url = f'{endpoint}/mcp/servers'
        headers = self.builder_headers(self.headers)

        # Only add Authorization header if token is provided
        if token:
            headers['Authorization'] = f'Bearer {token}'

        body = {
            'filter': filter_dict or {},
            'page_number': page_number,
            'page_size': page_size,
            'search': search
        }

        try:
            r = self.session.put(url, headers=headers, json=body)
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
            'mcp_server_list': mcp_server_list,
            'total_count': data.get('total_count', 0),
            'server_brief_list': server_brief_list
        }

    def get_mcp_server_operational(
            self,
            token: Optional[str] = None,
            endpoint: Optional[str] = None,
            convert_to_mcp_config: bool = True) -> Dict[str, Any]:
        """
        Get user-hosted MCP server list.

        Args:
            token: Authentication token (optional). If not provided, request will be made without authentication
            endpoint: API endpoint, defaults to MCP-specific endpoint (inherited from HubApi + /openapi/v1)
            convert_to_mcp_config: Whether to convert the result to MCP configuration format

        Returns:
            Dictionary containing MCP server list:
                - mcp_server_list: Detailed MCP server list
                - total_count: Total count
                - server_brief_list: Brief server information list

            If convert_to_mcp_config is True, returns:
                - mcpServers: Dictionary mapping server names to their configuration

        Raises:
            MCPApiRequestError: If the API request fails
            MCPApiResponseError: If the API response is invalid
        """
        if not endpoint:
            endpoint = self.endpoint

        url = f'{endpoint}/mcp/servers/operational'
        headers = self.builder_headers(self.headers)

        # Only add Authorization header if token is provided
        if token:
            headers['Authorization'] = f'Bearer {token}'

        try:
            r = self.session.get(url, headers=headers)
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
            'description': item.get('description', '')
        } for item in mcp_server_list]

        # Return original format if conversion is not requested
        if not convert_to_mcp_config:
            return {
                'mcp_server_list': mcp_server_list,
                'total_count': data.get('total_count', 0),
                'server_brief_list': server_brief_list
            }

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

        return {'mcpServers': mcp_servers}

    def get_mcp_server_special(
            self,
            server_id: str,
            token: Optional[str] = None,
            get_operational_url: bool = False,
            endpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Get special MCP server information.

        Args:
            server_id: ID of the MCP server
            token: Authentication token (optional)
            get_operational_url: Whether to get operational URLs
            endpoint: API endpoint, defaults to MCP-specific endpoint

        Returns:
            Dictionary containing server information

        Raises:
            ValueError: If server_id is empty
            MCPApiRequestError: If the API request fails
            MCPApiResponseError: If the API response is invalid
        """
        if not server_id:
            raise ValueError('server_id cannot be empty')

        if not endpoint:
            endpoint = self.endpoint

        url = f'{endpoint}/mcp/servers/{server_id}'
        headers = self.builder_headers(self.headers)

        # Only add Authorization header if token is provided
        if token:
            headers['Authorization'] = f'Bearer {token}'

        try:
            r = self.session.get(url, headers=headers)
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

        if get_operational_url:
            # Return operational URLs if requested
            operational_urls = data.get('operational_urls', [])
            return {'operational_urls': operational_urls, 'server_info': data}
        else:
            return data

    # def deploy_mcp_server(
    #     self,
    #     mcp_id: str,
    #     env_info: Optional[Dict[str, Any]] = None,
    #     auth_check: bool = False,
    #     token: Optional[str] = None,
    #     endpoint: Optional[str] = None
    # ) -> Dict[str, Any]:
    #     """
    #     Deploy an MCP server.

    #     Args:
    #         mcp_id: ID of the MCP server to deploy
    #         env_info: Environment information for deployment
    #         auth_check: Whether to perform authentication check
    #         token: Authentication token (optional)
    #         endpoint: API endpoint, defaults to MCP-specific endpoint

    #     Returns:
    #         Dictionary containing deployment result
    #             - ip: IP address of the MCP server
    #             - url: URL of the MCP server

    #     Raises:
    #         ValueError: If mcp_id is empty
    #         MCPApiRequestError: If the API request fails
    #         MCPApiResponseError: If the API response is invalid
    #     """
    #     if not mcp_id:
    #         raise ValueError("mcp_id cannot be empty")

    #     if not endpoint:
    #         endpoint = self.endpoint

    #     url = f"{endpoint}/mcp/servers/{mcp_id}/deploy"
    #     headers = self.builder_headers(self.headers)

    #     # Only add Authorization header if token is provided
    #     if token:
    #         headers["Authorization"] = f"Bearer {token}"

    #     body = {
    #         "env_info": env_info or {},
    #         "auth_check": auth_check
    #     }

    #     try:
    #         r = self.session.post(url, headers=headers, json=body)
    #         raise_for_http_status(r)
    #     except requests.exceptions.RequestException as e:
    #         logger.error(f"Failed to deploy MCP server {mcp_id}: {e}")
    #         raise MCPApiRequestError(f"Failed to deploy MCP server {mcp_id}: {e}") from e

    #     try:
    #         resp = r.json()
    #     except requests.exceptions.JSONDecodeError as e:
    #         logger.error(f"JSON parsing failed: {e}")
    #         logger.error(f"Response content: {r.text}")
    #         raise MCPApiResponseError(f"Invalid JSON response: {e}") from e

    #     return resp.get("data", {})

    # def undeploy_mcp_server(
    #     self,
    #     mcp_id: str,
    #     token: Optional[str] = None,
    #     endpoint: Optional[str] = None
    # ) -> Dict[str, Any]:
    #     """
    #     Undeploy an MCP server.

    #     Args:
    #         mcp_id: ID of the MCP server to undeploy
    #         token: Authentication token (optional)
    #         endpoint: API endpoint, defaults to MCP-specific endpoint

    #     Returns:
    #         Dictionary containing undeployment result

    #     Raises:
    #         ValueError: If mcp_id is empty
    #         MCPApiRequestError: If the API request fails
    #         MCPApiResponseError: If the API response is invalid
    #     """
    #     if not mcp_id:
    #         raise ValueError("mcp_id cannot be empty")

    #     if not endpoint:
    #         endpoint = self.endpoint

    #     url = f"{endpoint}/mcp/servers/{mcp_id}/undeploy"
    #     headers = self.builder_headers(self.headers)

    #     # Only add Authorization header if token is provided
    #     if token:
    #         headers["Authorization"] = f"Bearer {token}"

    #     try:
    #         r = self.session.delete(url, headers=headers)
    #         raise_for_http_status(r)
    #     except requests.exceptions.RequestException as e:
    #         logger.error(f"Failed to undeploy MCP server {mcp_id}: {e}")
    #         raise MCPApiRequestError(f"Failed to undeploy MCP server {mcp_id}: {e}") from e

    #     try:
    #         resp = r.json()
    #     except requests.exceptions.JSONDecodeError as e:
    #         logger.error(f"JSON parsing failed: {e}")
    #         logger.error(f"Response content: {r.text}")
    #         raise MCPApiResponseError(f"Invalid JSON response: {e}") from e

    #     return resp.get("data", {})
