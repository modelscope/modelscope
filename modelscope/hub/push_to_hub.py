# Copyright (c) Alibaba, Inc. and its affiliates.

import concurrent.futures
import os

from modelscope.hub.api import HubApi
from modelscope.hub.constants import ModelVisibility
from modelscope.utils.constant import DEFAULT_REPOSITORY_REVISION
from modelscope.utils.logger import get_logger

logger = get_logger()

_executor = concurrent.futures.ProcessPoolExecutor(max_workers=8)


def _api_push_to_hub(repo_name,
                     output_dir,
                     token,
                     private=True,
                     commit_message='',
                     tag=None,
                     source_repo='',
                     ignore_file_pattern=None,
                     revision=DEFAULT_REPOSITORY_REVISION):
    try:
        api = HubApi()
        api.login(token)
        api.push_model(
            repo_name,
            output_dir,
            visibility=ModelVisibility.PUBLIC
            if not private else ModelVisibility.PRIVATE,
            chinese_name=repo_name,
            commit_message=commit_message,
            tag=tag,
            original_model_id=source_repo,
            ignore_file_pattern=ignore_file_pattern,
            revision=revision)
        commit_message = commit_message or 'No commit message'
        logger.info(
            f'Successfully upload the model to {repo_name} with message: {commit_message}'
        )
        return True
    except Exception as e:
        logger.error(
            f'Error happens when uploading model {repo_name} with message: {commit_message}: {e}'
        )
        return False


def push_to_hub(repo_name,
                output_dir,
                token=None,
                private=True,
                retry=3,
                commit_message='',
                tag=None,
                source_repo='',
                ignore_file_pattern=None,
                revision=DEFAULT_REPOSITORY_REVISION):
    """
    Args:
        repo_name: The repo name for the modelhub repo
        output_dir: The local output_dir for the checkpoint
        token: The user api token, function will check the `MODELSCOPE_API_TOKEN` variable if this argument is None
        private: If is a private repo, default True
        retry: Retry times if something error in uploading, default 3
        commit_message: The commit message
        tag: The tag of this commit
        source_repo: The source repo (model id) which this model comes from
        ignore_file_pattern: The file pattern to be ignored in uploading.
        revision: The branch to commit to
    Returns:
        The boolean value to represent whether the model is uploaded.
    """
    if token is None:
        token = os.environ.get('MODELSCOPE_API_TOKEN')
    if ignore_file_pattern is None:
        ignore_file_pattern = os.environ.get('UPLOAD_IGNORE_FILE_PATTERN')
    assert repo_name is not None
    assert token is not None, 'Either pass in a token or to set `MODELSCOPE_API_TOKEN` in the environment variables.'
    assert os.path.isdir(output_dir)
    assert 'configuration.json' in os.listdir(output_dir) or 'configuration.yaml' in os.listdir(output_dir) \
           or 'configuration.yml' in os.listdir(output_dir)

    logger.info(
        f'Uploading {output_dir} to {repo_name} with message {commit_message}')
    for i in range(retry):
        if _api_push_to_hub(repo_name, output_dir, token, private,
                            commit_message, tag, source_repo,
                            ignore_file_pattern, revision):
            return True
    return False


def push_to_hub_async(repo_name,
                      output_dir,
                      token=None,
                      private=True,
                      commit_message='',
                      tag=None,
                      source_repo='',
                      ignore_file_pattern=None,
                      revision=DEFAULT_REPOSITORY_REVISION):
    """
    Args:
        repo_name: The repo name for the modelhub repo
        output_dir: The local output_dir for the checkpoint
        token: The user api token, function will check the `MODELSCOPE_API_TOKEN` variable if this argument is None
        private: If is a private repo, default True
        commit_message: The commit message
        tag: The tag of this commit
        source_repo: The source repo (model id) which this model comes from
        ignore_file_pattern: The file pattern to be ignored in uploading
        revision: The branch to commit to
    Returns:
        A handler to check the result and the status
    """
    if token is None:
        token = os.environ.get('MODELSCOPE_API_TOKEN')
    if ignore_file_pattern is None:
        ignore_file_pattern = os.environ.get('UPLOAD_IGNORE_FILE_PATTERN')
    assert repo_name is not None
    assert token is not None, 'Either pass in a token or to set `MODELSCOPE_API_TOKEN` in the environment variables.'
    assert os.path.isdir(output_dir)
    assert 'configuration.json' in os.listdir(output_dir) or 'configuration.yaml' in os.listdir(output_dir) \
           or 'configuration.yml' in os.listdir(output_dir)

    logger.info(
        f'Uploading {output_dir} to {repo_name} with message {commit_message}')
    return _executor.submit(_api_push_to_hub, repo_name, output_dir, token,
                            private, commit_message, tag, source_repo,
                            ignore_file_pattern, revision)
