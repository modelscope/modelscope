# Copyright (c) Alibaba, Inc. and its affiliates.
"""Agent-IDP API shim delegating to :mod:`modelscope_hub`.

Agent-IDP manages Ed25519 identities, OIDC discovery, and JWT issuance. It is
unrelated to Agent repository file transfer and deliberately owns no HTTP
transport: all protocol handling lives in ``modelscope_hub``'s OpenAPI client.
"""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Optional

from modelscope_hub.agent_idp import (generate_agent_key_pair,
                                      load_private_jwk,
                                      public_jwk_from_private,
                                      sign_agent_token_request,
                                      write_private_jwk)
from modelscope_hub.types import (AgentIdConfiguration, AgentIdentity,
                                  AgentIdentitySummary, AgentJWK, AgentToken,
                                  AgentTokenRecord, PagedResult)

from modelscope.hub.api import HubApi

__all__ = [
    'AgentIdpApi',
    'generate_agent_key_pair',
    'load_private_jwk',
    'public_jwk_from_private',
    'sign_agent_token_request',
    'write_private_jwk',
]


class AgentIdpApi(HubApi):
    """OpenAPI-first SDK surface for ModelScope Agent-IDP.

    The inherited ``HubApi`` remains the source of endpoint and ambient token
    configuration. Per-call tokens create a temporary hub client, matching the
    MCP delegation pattern without duplicating HTTP or signing logic upstream.
    """

    def _delegate(self, token: Optional[str] = None):
        if token:
            from modelscope_hub.api import HubApi as _HubApi
            return _HubApi(token=token, endpoint=self.endpoint)
        return self._api

    def create_agent_identity(self,
                              payload: Mapping[str, Any],
                              *,
                              token: Optional[str] = None) -> AgentIdentity:
        return self._delegate(token).create_agent_identity(payload)

    def get_agent_identity(self,
                           agent_id: str,
                           *,
                           token: Optional[str] = None) -> AgentIdentity:
        return self._delegate(token).get_agent_identity(agent_id)

    def update_agent_identity(
        self,
        agent_id: str,
        payload: Mapping[str, Any],
        *,
        token: Optional[str] = None,
    ) -> AgentIdentity:
        return self._delegate(token).update_agent_identity(agent_id, payload)

    def delete_agent_identity(self,
                              agent_id: str,
                              *,
                              token: Optional[str] = None) -> dict:
        return self._delegate(token).delete_agent_identity(agent_id)

    def reset_agent_key_pair(
        self,
        agent_id: str,
        payload: Mapping[str, Any],
        *,
        token: Optional[str] = None,
    ) -> AgentIdentity:
        return self._delegate(token).reset_agent_key_pair(agent_id, payload)

    def pause_agent(self,
                    agent_id: str,
                    *,
                    paused: bool,
                    token: Optional[str] = None) -> dict:
        return self._delegate(token).pause_agent(agent_id, paused=paused)

    def list_agent_token_records(
        self,
        agent_id: str,
        *,
        page: int = 1,
        page_size: int = 20,
        token: Optional[str] = None,
    ) -> PagedResult[AgentTokenRecord]:
        return self._delegate(token).list_agent_token_records(
            agent_id, page=page, page_size=page_size)

    def list_user_agent_identities(
        self,
        username: str,
        *,
        status: str | None = None,
        page: int = 1,
        page_size: int = 20,
        token: Optional[str] = None,
    ) -> PagedResult[AgentIdentitySummary]:
        return self._delegate(token).list_user_agent_identities(
            username,
            status=status,
            page=page,
            page_size=page_size,
        )

    def issue_agent_token(self, payload: Mapping[str, Any]) -> AgentToken:
        """Exchange a signed Agent-IDP request; this OpenAPI operation is anonymous."""
        return self._api.issue_agent_token(payload)

    def issue_agent_token_with_private_key(
        self,
        private_jwk: AgentJWK | Mapping[str, Any],
        *,
        agent_id: str,
        audience: str,
        timestamp: int | None = None,
    ) -> AgentToken:
        return self._api.issue_agent_token_with_private_key(
            private_jwk,
            agent_id=agent_id,
            audience=audience,
            timestamp=timestamp,
        )

    def get_agent_id_configuration(self) -> AgentIdConfiguration:
        """Return anonymous Agent-IDP OIDC discovery metadata."""
        return self._api.get_agent_id_configuration()

    def get_agent_id_jwks(self) -> list[AgentJWK]:
        """Return anonymous Agent-IDP JWT verification keys."""
        return self._api.get_agent_id_jwks()
