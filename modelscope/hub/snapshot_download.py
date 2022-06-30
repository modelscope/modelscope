import os
import tempfile
from glob import glob
from pathlib import Path
from typing import Dict, Optional, Union

from modelscope.utils.logger import get_logger
from .api import HubApi, ModelScopeConfig
from .constants import DEFAULT_MODELSCOPE_GROUP, MODEL_ID_SEPARATOR
from .errors import NotExistError, RequestError, raise_on_error
from .file_download import (get_file_download_url, http_get_file,
                            http_user_agent)
from .utils.caching import ModelFileSystemCache
from .utils.utils import get_cache_dir, model_id_to_group_owner_name

logger = get_logger()


def snapshot_download(model_id: str,
                      revision: Optional[str] = 'master',
                      cache_dir: Union[str, Path, None] = None,
                      user_agent: Optional[Union[Dict, str]] = None,
                      local_files_only: Optional[bool] = False) -> str:
    """Download all files of a repo.
    Downloads a whole snapshot of a repo's files at the specified revision. This
    is useful when you want all files from a repo, because you don't know which
    ones you will need a priori. All files are nested inside a folder in order
    to keep their actual filename relative to that folder.

    An alternative would be to just clone a repo but this would require that the
    user always has git and git-lfs installed, and properly configured.
    Args:
        model_id (`str`):
            A user or an organization name and a repo name separated by a `/`.
        revision (`str`, *optional*):
            An optional Git revision id which can be a branch name, a tag, or a
            commit hash. NOTE: currently only branch and tag name is supported
        cache_dir (`str`, `Path`, *optional*):
            Path to the folder where cached files are stored.
        user_agent (`str`, `dict`, *optional*):
            The user-agent info in the form of a dictionary or a string.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, avoid downloading the file and return the path to the
            local cached file if it exists.
    Returns:
        Local folder path (string) of repo snapshot

    <Tip>
    Raises the following errors:
    - [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)
      if `use_auth_token=True` and the token cannot be found.
    - [`OSError`](https://docs.python.org/3/library/exceptions.html#OSError) if
      ETag cannot be determined.
    - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
      if some parameter value is invalid
    </Tip>
    """

    if cache_dir is None:
        cache_dir = get_cache_dir()
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    group_or_owner, name = model_id_to_group_owner_name(model_id)

    cache = ModelFileSystemCache(cache_dir, group_or_owner, name)
    if local_files_only:
        if len(cache.cached_files) == 0:
            raise ValueError(
                'Cannot find the requested files in the cached path and outgoing'
                ' traffic has been disabled. To enable model look-ups and downloads'
                " online, set 'local_files_only' to False.")
        logger.warn('We can not confirm the cached file is for revision: %s'
                    % revision)
        return cache.get_root_location(
        )  # we can not confirm the cached file is for snapshot 'revision'
    else:
        # make headers
        headers = {'user-agent': http_user_agent(user_agent=user_agent, )}
        _api = HubApi()
        cookies = ModelScopeConfig.get_cookies()
        # get file list from model repo
        branches, tags = _api.get_model_branches_and_tags(
            model_id, use_cookies=False if cookies is None else cookies)
        if revision not in branches and revision not in tags:
            raise NotExistError('The specified branch or tag : %s not exist!'
                                % revision)

        model_files = _api.get_model_files(
            model_id=model_id,
            revision=revision,
            recursive=True,
            use_cookies=False if cookies is None else cookies,
            is_snapshot=True)

        for model_file in model_files:
            if model_file['Type'] == 'tree':
                continue
            # check model_file is exist in cache, if exist, skip download, otherwise download
            if cache.exists(model_file):
                logger.info(
                    'The specified file is in cache, skip downloading!')
                continue

            # get download url
            url = get_file_download_url(
                model_id=model_id,
                file_path=model_file['Path'],
                revision=revision)

            # First download to /tmp
            http_get_file(
                url=url,
                local_dir=tempfile.gettempdir(),
                file_name=model_file['Name'],
                headers=headers,
                cookies=cookies)
            # put file to cache
            cache.put_file(
                model_file,
                os.path.join(tempfile.gettempdir(), model_file['Name']))

        return os.path.join(cache.get_root_location())
