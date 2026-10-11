# Copyright (c) Alibaba, Inc. and its affiliates.
"""Import-compatible Agent-IDP CLI shim.

The four ModelScope console scripts are owned by modelscope-hub. This module
keeps direct imports from the umbrella SDK working without registering a
second CLI plugin or implementation.
"""
from modelscope_hub.cli.agent_idp import AgentIdpCommand


def subparser_func(args):
    """Return the hub-owned Agent-IDP command for a parsed namespace."""
    return AgentIdpCommand(args)


__all__ = ['AgentIdpCommand', 'subparser_func']
