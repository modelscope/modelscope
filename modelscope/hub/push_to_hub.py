# Copyright (c) Alibaba, Inc. and its affiliates.

import concurrent.futures
import os

from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility
from modelscope.hub.errors import NotExistError
from modelscope.utils.logger import get_logger

logger = get_logger()

_executor = concurrent.futures.ProcessPoolExecutor(max_workers=8)


def _api_push_to_hub(repo_name,
                     output_dir,
                     token,
                     private=True,
                     commit_message='',
                     source_repo=''):
    api = HubApi()
    api.login(token)
    api.push_model(
        repo_name,
        output_dir,
        visibility=ModelVisibility.PUBLIC
        if not private else ModelVisibility.PRIVATE,
        chinese_name=repo_name,
        commit_message=commit_message,
        original_model_id=source_repo)
    commit_message = commit_message or 'No commit message'
    logger.info(
        f'Successfully upload the model to {repo_name} with message: {commit_message}'
    )


def push_to_hub(repo_name,
                output_dir,
                token=None,
                private=True,
                retry=3,
                commit_message='',
                source_repo='',
                async_upload=False):
    """
    Args:
        repo_name: The repo name for the modelhub repo
        output_dir: The local output_dir for the checkpoint
        token: The user api token, function will check the `MODELSCOPE_API_TOKEN` variable if this argument is None
        private: If is a private repo, default True
        retry: Retry times if something error in uploading, default 3
        commit_message: The commit message
        source_repo: The source repo (model id) which this model comes from
        async_upload: Upload async, if True, the `retry` parameter will have no affection

    Returns:
        A handler if async_upload is True, else None
    """
    if token is None:
        token = os.environ.get('MODELSCOPE_API_TOKEN')
    assert token is not None, 'Either pass in a token or to set `MODELSCOPE_API_TOKEN` in the environment variables.'
    assert os.path.isdir(output_dir)
    assert 'configuration.json' in os.listdir(output_dir) or 'configuration.yaml' in os.listdir(output_dir) \
           or 'configuration.yml' in os.listdir(output_dir)

    logger.info(
        f'Uploading {output_dir} to {repo_name} with message {commit_message}')
    if not async_upload:
        for i in range(retry):
            try:
                _api_push_to_hub(repo_name, output_dir, token, private,
                                 commit_message, source_repo)
            except Exception as e:
                logger.info(f'Error happens when uploading model: {e}')
                continue
            break
    else:
        return _executor.submit(_api_push_to_hub, repo_name, output_dir, token,
                                private, commit_message, source_repo)
