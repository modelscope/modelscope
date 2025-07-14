# Copyright (c) Alibaba, Inc. and its affiliates.
"""
configuration management for MCP servers.
"""

import logging
from typing import Any, Dict

import json

from modelscope.hub.modelscope_mcp.types import validate_mcp_config
from modelscope.hub.modelscope_mcp.utils import (MCPConfigError,
                                                 merge_mcp_configs)

logger = logging.getLogger(__name__)


def load_config(file_path: str) -> Dict[str, Any]:
    """Load MCP configuration from JSON file.

    Args:
        file_path: Path to the JSON configuration file.

    Returns:
        Dict containing MCP configuration.

    Raises:
        FileNotFoundError: If configuration file doesn't exist.
        JSONDecodeError: If configuration file is invalid JSON.
        MCPConfigError: If configuration is invalid.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        validate_mcp_config(config)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f'Configuration file not found: {file_path}')
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f'Invalid JSON in {file_path}: {e}', e.doc,
                                   e.pos)
    except Exception as e:
        raise MCPConfigError(f'Error reading configuration: {e}')


def merge_configs(local_config: Dict[str, Any],
                  server_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge local and server configurations.

    Args:
        local_config: Local MCP configuration
        server_config: Server MCP configuration

    Returns:
        Merged configuration dictionary
    """
    return merge_mcp_configs([local_config, server_config])


def create_default_config(file_path: str) -> None:
    """Create a default MCP configuration file.

    Args:
        file_path: Path where to create the configuration file
    """
    default_config = {
        'mcpServers': {
            'filesystem': {
                'command': 'npx',
                'args': ['-y', '@modelcontextprotocol/server-filesystem', '.']
            }
        }
    }

    from pathlib import Path
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=2, ensure_ascii=False)

    logger.info(f'Created default configuration at: {file_path}')
