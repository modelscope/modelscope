# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Dict, Optional, Union
from urllib.parse import urlparse

from modelscope.hub.api import HubApi, ModelScopeConfig
from modelscope.hub.constants import FILE_HASH
from modelscope.hub.git import GitCommandWrapper
from modelscope.hub.utils.caching import ModelFileSystemCache
from modelscope.hub.utils.utils import compute_hash
from modelscope.utils.logger import get_logger

logger = get_logger()


def check_local_model_is_latest(
    model_root_path: str,
    user_agent: Optional[Union[Dict, str]] = None,
):
    """Check local model repo is latest.
    Check local model repo is same as hub latest version.
    """
    model_cache = None
    # download with git
    if os.path.exists(os.path.join(model_root_path, '.git')):
        git_cmd_wrapper = GitCommandWrapper()
        git_url = git_cmd_wrapper.get_repo_remote_url(model_root_path)
        if git_url.endswith('.git'):
            git_url = git_url[:-4]
        u_parse = urlparse(git_url)
        model_id = u_parse.path[1:]
    else:  # snapshot_download
        model_cache = ModelFileSystemCache(model_root_path)
        model_id = model_cache.get_model_id()

    try:
        # make headers
        headers = {
            'user-agent':
            ModelScopeConfig.get_user_agent(user_agent=user_agent, )
        }
        cookies = ModelScopeConfig.get_cookies()

        snapshot_header = headers if 'CI_TEST' in os.environ else {
            **headers,
            **{
                'Snapshot': 'True'
            }
        }
        _api = HubApi()
        try:
            _, revisions = _api.get_model_branches_and_tags(
                model_id=model_id, use_cookies=cookies)
            if len(revisions) > 0:
                latest_revision = revisions[0]
            else:
                latest_revision = 'master'
        except:  # noqa: E722
            latest_revision = 'master'

        model_files = _api.get_model_files(
            model_id=model_id,
            revision=latest_revision,
            recursive=True,
            headers=snapshot_header,
            use_cookies=cookies,
        )
        for model_file in model_files:
            if model_file['Type'] == 'tree':
                continue
            # check model_file updated
            if model_cache is not None:
                if model_cache.exists(model_file):
                    continue
                else:
                    logger.info(
                        'Model is updated from modelscope hub, you can verify from http://www.modelscope.cn.'
                    )
                    break
            else:
                if FILE_HASH in model_file:
                    local_file_hash = compute_hash(
                        os.path.join(model_root_path, model_file['Path']))
                    if local_file_hash == model_file[FILE_HASH]:
                        continue
                    else:
                        logger.info(
                            'Model is updated from modelscope hub, you can verify from http://www.modelscope.cn.'
                        )
                        break
    except:  # noqa: E722
        pass  # ignore
