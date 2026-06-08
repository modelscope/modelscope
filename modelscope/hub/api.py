# Copyright (c) Alibaba, Inc. and its affiliates.
"""Hub API — shim delegating to modelscope_hub.

This module preserves the legacy ``modelscope.hub.api`` public surface
(``HubApi``, ``ModelScopeConfig``, ``model_id_to_group_owner_name`` and a few
response-field constants) by delegating to the ``modelscope_hub`` package.

Single responsibility: thin compatibility layer. All real logic lives in
``modelscope_hub.compat.LegacyHubApi`` and ``modelscope_hub.config.HubConfig``.
"""
from __future__ import annotations

import os
import platform
from os.path import expanduser
from typing import Dict, Optional, Tuple, Union

from modelscope_hub.compat import LegacyHubApi as _LegacyHubApi
from modelscope_hub.config import get_default_config

from modelscope.hub.constants import (API_HTTP_CLIENT_MAX_RETRIES,
                                      API_HTTP_CLIENT_TIMEOUT,
                                      API_RESPONSE_FIELD_DATA,
                                      API_RESPONSE_FIELD_EMAIL,
                                      API_RESPONSE_FIELD_GIT_ACCESS_TOKEN,
                                      API_RESPONSE_FIELD_MESSAGE,
                                      API_RESPONSE_FIELD_USERNAME,
                                      MODELSCOPE_CLOUD_ENVIRONMENT,
                                      MODELSCOPE_CLOUD_USERNAME,
                                      MODELSCOPE_CREDENTIALS_PATH)
from modelscope.hub.utils.utils import model_id_to_group_owner_name
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = [
    'HubApi',
    'ModelScopeConfig',
    'model_id_to_group_owner_name',
    'API_RESPONSE_FIELD_DATA',
    'API_RESPONSE_FIELD_MESSAGE',
    'API_RESPONSE_FIELD_USERNAME',
    'API_RESPONSE_FIELD_EMAIL',
    'API_RESPONSE_FIELD_GIT_ACCESS_TOKEN',
]


class HubApi(_LegacyHubApi):
    """ModelScope Hub API — delegates to ``modelscope_hub``.

    Maintains backward compatibility with the legacy ``HubApi`` interface;
    method behaviour is inherited from
    :class:`modelscope_hub.compat.LegacyHubApi`.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        timeout: int = API_HTTP_CLIENT_TIMEOUT,
        max_retries: int = API_HTTP_CLIENT_MAX_RETRIES,
        token: Optional[str] = None,
    ) -> None:
        super().__init__(endpoint=endpoint, token=token)
        # Preserved for callers that historically read these attributes.
        self.endpoint = self._endpoint or self._api._config.endpoint
        self.token = token
        self.timeout = timeout
        self.max_retries = max_retries
        self.headers = {'user-agent': ModelScopeConfig.get_user_agent()}

        # If non-default timeout/max_retries were provided, eagerly construct
        # the internal LegacyClient so they actually take effect on the wire.
        if (timeout != API_HTTP_CLIENT_TIMEOUT
                or max_retries != API_HTTP_CLIENT_MAX_RETRIES):
            from modelscope_hub._legacy_api import LegacyClient
            from modelscope_hub.utils import build_user_agent
            cfg = self._api._config
            self._api._legacy = LegacyClient(
                token=cfg.token,
                endpoint=cfg.endpoint,
                timeout=timeout,
                max_retries=max_retries,
                user_agent=build_user_agent(cfg.get_session_id()),
            )

    # ------------------------------------------------------------------
    # Legacy method shims missing from LegacyHubApi
    # ------------------------------------------------------------------
    def create_model(self, model_id: str, **kwargs) -> None:
        """Create a model repo — delegates to ``create_repo`` (model type)."""
        return self.create_repo(model_id, repo_type='model', **kwargs)

    def get_model_url(self, model_id: str) -> str:
        """Return the model page URL ``{endpoint}/{model_id}``."""
        return f'{self.endpoint}/{model_id}'

    def upload_folder(self, repo_id: str, folder_path=None, **kwargs):
        """Upload a folder — delegates to internal ``HubApi.upload_folder``."""
        from modelscope_hub.api import HubApi as _NewHubApi
        repo_type = kwargs.pop('repo_type', None) or 'model'
        token = kwargs.pop('token', None)
        api = self._api
        if token and token != self._api._config.token:
            api = _NewHubApi(token=token, endpoint=self._api._config.endpoint)
        return api.upload_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=folder_path,
            **kwargs,
        )

    def upload_file(self, repo_id: str = None, path_or_fileobj=None,
                    path_in_repo: str = None, **kwargs):
        """Upload a file — delegates to internal ``HubApi.upload_file``."""
        from modelscope_hub.api import HubApi as _NewHubApi
        repo_type = kwargs.pop('repo_type', None) or 'model'
        token = kwargs.pop('token', None)
        api = self._api
        if token and token != self._api._config.token:
            api = _NewHubApi(token=token, endpoint=self._api._config.endpoint)
        return api.upload_file(
            repo_id=repo_id,
            repo_type=repo_type,
            path_or_fileobj=path_or_fileobj,
            path_in_repo=path_in_repo,
            **kwargs,
        )

    @property
    def _prepare_upload_folder(self):
        """Expose UploadManager._prepare_upload_folder for monkey-patching."""
        return self._api.uploader._prepare_upload_folder

    @_prepare_upload_folder.setter
    def _prepare_upload_folder(self, value):
        """Allow CommitScheduler to monkey-patch ``_prepare_upload_folder``."""
        self._api.uploader._prepare_upload_folder = value

    def __getattr__(self, name: str):
        """Transparent proxy to the internal ``modelscope_hub.HubApi``.

        Only invoked when normal attribute lookup (instance dict, class
        hierarchy including :class:`LegacyHubApi`) fails. Private names are
        excluded to avoid recursion during ``__init__``.
        """
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            inner = object.__getattribute__(self, '_api')
        except AttributeError as exc:
            raise AttributeError(name) from exc
        try:
            return getattr(inner, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from None


class ModelScopeConfig:
    """Configuration manager — delegates to ``modelscope_hub.HubConfig``.

    Preserves the static-method interface used throughout the legacy
    codebase. Class-level attributes are kept for callers that read them
    directly (e.g. ``ModelScopeConfig.path_credential``).
    """

    path_credential = expanduser(MODELSCOPE_CREDENTIALS_PATH)
    COOKIES_FILE_NAME = 'cookies'
    GIT_TOKEN_FILE_NAME = 'git_token'
    USER_INFO_FILE_NAME = 'user'
    USER_SESSION_ID_FILE_NAME = 'session'
    cookie_expired_warning = False

    @staticmethod
    def make_sure_credential_path_exist() -> None:
        """Ensure the credentials directory exists."""
        os.makedirs(ModelScopeConfig.path_credential, exist_ok=True)
        get_default_config().ensure_dirs()

    @staticmethod
    def save_cookies(cookies) -> None:
        """Persist cookies to disk."""
        get_default_config().save_cookies(cookies)

    @staticmethod
    def get_cookies():
        """Load persisted cookies, returning ``None`` if absent or expired."""
        cookies = get_default_config().load_cookies()
        if cookies is None and not ModelScopeConfig.cookie_expired_warning:
            ModelScopeConfig.cookie_expired_warning = True
        return cookies

    @staticmethod
    def get_user_session_id() -> str:
        """Return a stable session ID used in the user-agent header."""
        return get_default_config().get_session_id()

    @staticmethod
    def save_git_token(git_token: str) -> None:
        """Persist a git access token."""
        get_default_config().save_git_token(git_token)

    @staticmethod
    def save_token(token: str) -> None:
        """Deprecated: use :meth:`save_git_token` instead."""
        import warnings
        warnings.warn(
            'ModelScopeConfig.save_token() is deprecated, '
            'use ModelScopeConfig.save_git_token() instead.',
            DeprecationWarning, stacklevel=2)
        ModelScopeConfig.save_git_token(token)

    @staticmethod
    def save_user_info(user_name: str, user_email: str) -> None:
        """Persist ``user_name:user_email`` to the credentials directory."""
        get_default_config().save_user_info(user_name, user_email)

    @staticmethod
    def get_user_info() -> Tuple[Optional[str], Optional[str]]:
        """Return ``(username, email)`` previously saved, or ``(None, None)``."""
        path = get_default_config().credentials_dir / ModelScopeConfig.USER_INFO_FILE_NAME
        try:
            info = path.read_text(encoding='utf-8')
        except (FileNotFoundError, OSError):
            return None, None
        parts = info.split(':', 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return None, None

    @staticmethod
    def get_git_token() -> Optional[str]:
        """Return the persisted git access token, or ``None`` if not set."""
        return get_default_config().load_git_token()

    @staticmethod
    def get_token() -> Optional[str]:
        """Deprecated: use :meth:`get_git_token` instead."""
        import warnings
        warnings.warn(
            'ModelScopeConfig.get_token() is deprecated, '
            'use ModelScopeConfig.get_git_token() instead.',
            DeprecationWarning, stacklevel=2)
        return ModelScopeConfig.get_git_token()

    @staticmethod
    def get_user_agent(user_agent: Union[Dict, str, None] = None) -> str:
        """Build a user-agent string carrying SDK version and telemetry."""
        env = os.environ.get(MODELSCOPE_CLOUD_ENVIRONMENT, 'custom')
        user_name = os.environ.get(MODELSCOPE_CLOUD_USERNAME, 'unknown')

        from modelscope import __version__
        ua = (
            f'modelscope/{__version__}; '
            f'python/{platform.python_version()}; '
            f'session_id/{ModelScopeConfig.get_user_session_id()}; '
            f'platform/{platform.platform()}; '
            f'processor/{platform.processor()}; '
            f'env/{env}; '
            f'user/{user_name}'
        )
        if isinstance(user_agent, dict):
            ua += '; ' + '; '.join(f'{k}/{v}' for k, v in user_agent.items())
        elif isinstance(user_agent, str):
            ua += '; ' + user_agent
        return ua
