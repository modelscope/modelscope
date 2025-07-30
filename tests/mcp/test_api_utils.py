# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest
from unittest.mock import Mock

import json
import requests

from modelscope.mcp.api import MCPApi, MCPApiResponseError
from modelscope.utils.logger import get_logger

logger = get_logger()


class ApiUtilsTest(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.api = MCPApi()

    def test_get_server_name_from_id(self):
        """Test _get_server_name_from_id function."""
        # Test @group/server format
        self.assertEqual(
            MCPApi._get_server_name_from_id('@demo/weather-tool'),
            'weather-tool')
        # Test plain server name
        self.assertEqual(
            MCPApi._get_server_name_from_id('weather-tool'), 'weather-tool')
        # Test @server format
        self.assertEqual(
            MCPApi._get_server_name_from_id('@weather'), 'weather')

    def test_handle_response_success(self):
        """Test _handle_response with valid response."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'data': {
                'servers': [],
                'total_count': 0
            },
            'code': 200
        }

        result = self.api._handle_response(mock_response)
        self.assertEqual(result, {'servers': [], 'total_count': 0})

    def test_handle_response_no_data(self):
        """Test _handle_response when no data field."""
        mock_response = Mock()
        mock_response.json.return_value = {'code': 200}

        result = self.api._handle_response(mock_response)
        self.assertEqual(result, {})

    def test_handle_response_invalid_json(self):
        """Test _handle_response with invalid JSON."""
        mock_response = Mock()
        mock_response.json.side_effect = requests.exceptions.JSONDecodeError(
            'Invalid JSON', '', 0)
        mock_response.text = 'invalid json'

        with self.assertRaises(MCPApiResponseError):
            self.api._handle_response(mock_response)


if __name__ == '__main__':
    unittest.main()
