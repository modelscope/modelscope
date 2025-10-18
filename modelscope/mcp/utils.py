# Copyright (c) Alibaba, Inc. and its affiliates.
"""
MCP Utilities

This module contains utility functions and classes for MCP request handling,
including adapters for different LLM frameworks and MCP request building.

Note: Response processing is not needed as MCPClient.call_tool() returns ready-to-use strings.
"""

from typing import Any, Dict

import json

from modelscope.utils.logger import get_logger

logger = get_logger(__name__)


class MCPRequestAdapter:
    """Universal request adapter: Extract tool name and arguments"""

    @staticmethod
    def normalize_function_call(
            function_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Universal function call normalization

        Args:
            function_call: Original function call

        Returns:
            Standardized format: {"tool_name": str, "tool_args": dict}
        """
        try:
            return MCPRequestAdapter._universal_extract(function_call)
        except Exception as e:
            logger.error(f'Failed to normalize function call: {e}')
            raise ValueError(f'Function call normalization failed: {e}') from e

    @staticmethod
    def _universal_extract(call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Universal extraction: Try common patterns until one works

        This approach is much simpler - we just try different extraction patterns
        in order of likelihood until we find tool_name and tool_args.
        """
        if not isinstance(call, dict):
            raise ValueError('Function call must be a dictionary')

        # Pattern 1: Direct name/arguments (OpenAI style)
        tool_name = call.get('name')
        tool_args = call.get('arguments')
        if tool_name and tool_args is not None:
            return {
                'tool_name': tool_name,
                'tool_args': MCPRequestAdapter._parse_args(tool_args)
            }

        # Pattern 2: Direct name/input (LangChain style)
        tool_name = call.get('name')
        tool_args = call.get('input')
        if tool_name and tool_args is not None:
            return {
                'tool_name': tool_name,
                'tool_args': MCPRequestAdapter._parse_args(tool_args)
            }

        # Pattern 3: Nested function object (OpenAI tool_calls, Transformers)
        if 'function' in call and isinstance(call['function'], dict):
            func_obj = call['function']
            tool_name = func_obj.get('name')
            tool_args = func_obj.get('arguments')
            if tool_name:
                return {
                    'tool_name': tool_name,
                    'tool_args': MCPRequestAdapter._parse_args(tool_args)
                }

        # Pattern 4: tool_calls array (Mistral style)
        if 'tool_calls' in call and isinstance(
                call['tool_calls'], list) and len(call['tool_calls']) > 0:
            first_call = call['tool_calls'][0]
            if 'function' in first_call and isinstance(first_call['function'],
                                                       dict):
                func_obj = first_call['function']
                tool_name = func_obj.get('name')
                tool_args = func_obj.get('arguments')
                if tool_name:
                    return {
                        'tool_name': tool_name,
                        'tool_args': MCPRequestAdapter._parse_args(tool_args)
                    }

        # Pattern 5: Default format (tool_name/params)
        tool_name = call.get('tool_name')
        if tool_name:
            tool_args = call.get('tool_args') or call.get('params') or {}
            return {
                'tool_name': tool_name,
                'tool_args': MCPRequestAdapter._parse_args(tool_args)
            }

        # Pattern 6: Fallback - any field that looks like a name
        for name_field in ['name', 'tool', 'function_name', 'method']:
            if name_field in call:
                tool_name = call[name_field]
                # Try to find corresponding args field
                for args_field in [
                        'arguments', 'input', 'params', 'args', 'parameters'
                ]:
                    if args_field in call:
                        tool_args = call[args_field]
                        return {
                            'tool_name': tool_name,
                            'tool_args':
                            MCPRequestAdapter._parse_args(tool_args)
                        }
                # No args found, use empty dict
                return {'tool_name': tool_name, 'tool_args': {}}

        # If we get here, we couldn't extract anything useful
        raise ValueError(
            f'Could not extract tool name from function call: {call}')

    @staticmethod
    def _parse_args(args: Any) -> dict:
        """Parse arguments - handle JSON strings, dicts, or other formats"""
        if args is None:
            return {}

        if isinstance(args, dict):
            return args

        if isinstance(args, str):
            # Try to parse as JSON
            try:
                parsed = json.loads(args)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                # If not valid JSON, wrap in a dict
                return {'value': args}

        if isinstance(args, (list, tuple)):
            # Convert list/tuple to dict with numeric keys
            return {str(i): v for i, v in enumerate(args)}

        # For other types, wrap in a dict
        return {'value': args}
