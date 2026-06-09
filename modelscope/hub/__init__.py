"""modelscope.hub — shim layer delegating to modelscope_hub."""

import os as _os
from pathlib import Path as _Path

from modelscope_hub import HubConfig as _HubConfig
from modelscope_hub import get_default_config as _get_default_config
from modelscope_hub import set_default_config as _set_default_config

from .callback import ProgressCallback, TqdmCallback
from .commit_scheduler import CommitScheduler
from .snapshot_download import snapshot_download


def _sync_config() -> None:
    """Bridge legacy env vars that modelscope_hub does not natively recognize."""
    # MODELSCOPE_CACHE is already handled by HubConfig; only sync
    # if a non-standard alias is set.
    legacy_cache = _os.environ.get('MS_CACHE_HOME')
    if legacy_cache and not _os.environ.get('MODELSCOPE_CACHE'):
        _set_default_config(_HubConfig(cache_dir=legacy_cache))

    # Bridge MODELSCOPE_CREDENTIALS_PATH → HubConfig.config_dir so credential
    # lookup honours the legacy override.
    creds_path = _os.environ.get('MODELSCOPE_CREDENTIALS_PATH')
    if creds_path:
        resolved = _Path(creds_path).expanduser().resolve()
        # Legacy convention may point at either the credentials directory
        # (e.g. ``~/.modelscope``) or at a credentials file inside it; the
        # new HubConfig always expects the directory. Treat the path as a
        # file when it exists as one, falling back to the legacy
        # ``credentials`` filename heuristic for paths that do not yet
        # exist on disk.
        is_file = resolved.is_file() or (not resolved.exists()
                                         and resolved.name == 'credentials')
        config_dir = resolved.parent if is_file else resolved
        cfg = _get_default_config()
        if cfg.config_dir != config_dir:
            cfg.config_dir = config_dir


_sync_config()

__all__ = [
    'CommitScheduler',
    'ProgressCallback',
    'TqdmCallback',
    'snapshot_download',
]
