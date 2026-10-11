# Copyright (c) Alibaba, Inc. and its affiliates.
"""Delegation tests for the upstream Agent-IDP API shim."""
from unittest import mock

from modelscope.hub.agent_idp_api import AgentIdpApi


class TestAgentIdpApi:

    def test_authenticated_operations_delegate_without_http(self):
        api = AgentIdpApi()
        delegated = mock.MagicMock()
        api._api = delegated

        api.create_agent_identity({'agent_name': 'builder', 'public_key': {}})
        api.get_agent_identity('agent-1')
        api.update_agent_identity('agent-1', {'description': 'updated'})
        api.delete_agent_identity('agent-1')
        api.reset_agent_key_pair('agent-1', {'public_key': {}})
        api.pause_agent('agent-1', paused=True)
        api.list_agent_token_records('agent-1', page=2, page_size=10)
        api.list_user_agent_identities(
            'alice', status='paused', page=3, page_size=20)

        delegated.create_agent_identity.assert_called_once_with({
            'agent_name': 'builder',
            'public_key': {}
        })
        delegated.get_agent_identity.assert_called_once_with('agent-1')
        delegated.update_agent_identity.assert_called_once_with(
            'agent-1', {'description': 'updated'})
        delegated.delete_agent_identity.assert_called_once_with('agent-1')
        delegated.reset_agent_key_pair.assert_called_once_with(
            'agent-1', {'public_key': {}})
        delegated.pause_agent.assert_called_once_with('agent-1', paused=True)
        delegated.list_agent_token_records.assert_called_once_with(
            'agent-1', page=2, page_size=10)
        delegated.list_user_agent_identities.assert_called_once_with(
            'alice', status='paused', page=3, page_size=20)

    def test_public_operations_delegate_without_ambient_token_override(self):
        api = AgentIdpApi()
        delegated = mock.MagicMock()
        api._api = delegated
        payload = {
            'agent_id': 'agent-1',
            'kid': 'key-1',
            'audience': 'hub',
            'timestamp': 1,
            'signature': 'sig'
        }

        api.issue_agent_token(payload)
        api.issue_agent_token_with_private_key({'private': 'jwk'},
                                               agent_id='agent-1',
                                               audience='hub',
                                               timestamp=1)
        api.get_agent_id_configuration()
        api.get_agent_id_jwks()

        delegated.issue_agent_token.assert_called_once_with(payload)
        delegated.issue_agent_token_with_private_key.assert_called_once_with(
            {'private': 'jwk'},
            agent_id='agent-1',
            audience='hub',
            timestamp=1)
        delegated.get_agent_id_configuration.assert_called_once_with()
        delegated.get_agent_id_jwks.assert_called_once_with()

    def test_per_call_token_constructs_a_single_hub_client(self):
        api = AgentIdpApi(endpoint='https://example.test')
        delegated = mock.MagicMock()
        with mock.patch(
                'modelscope_hub.api.HubApi',
                return_value=delegated) as hub_api:
            api.get_agent_identity('agent-1', token='ms-write')
        hub_api.assert_called_once_with(
            token='ms-write', endpoint='https://example.test')
        delegated.get_agent_identity.assert_called_once_with('agent-1')
