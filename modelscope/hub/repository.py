# Copyright (c) Alibaba, Inc. and its affiliates.
"""Repository — shim delegating to ``modelscope_hub`` for Git operations.

Preserves the legacy :class:`Repository` and :class:`DatasetRepository`
constructors (auto-clone-on-init) and methods (``push``, ``pull``,
``tag``, ``tag_and_push``, ``add_lfs_type``) used by callers that
pre-date the SDK refactor.
"""
from __future__ import annotations

import os
import warnings
from typing import Optional

from modelscope.hub.errors import (GitError, InvalidParameter,
                                   NotLoginException)
from modelscope.utils.constant import (DEFAULT_DATASET_REVISION,
                                       DEFAULT_REPOSITORY_REVISION,
                                       MASTER_MODEL_BRANCH)
from modelscope.utils.logger import get_logger
from .git import GitCommandWrapper
from .utils.utils import get_endpoint

logger = get_logger()

__all__ = ['Repository', 'DatasetRepository']


def _resolve_token(auth_token: Optional[str]) -> Optional[str]:
    if auth_token:
        return auth_token
    from modelscope.hub.api import ModelScopeConfig
    return ModelScopeConfig.get_token()


def _clone_if_needed(git_wrapper: GitCommandWrapper,
                     base_dir: str,
                     repo_name: str,
                     repo_dir: str,
                     url: str,
                     token: Optional[str],
                     revision: Optional[str]) -> bool:
    """Clone *url* into *repo_dir* unless it's already that working copy.

    Returns ``True`` if a clone was performed, ``False`` if skipped.
    """
    os.makedirs(repo_dir, exist_ok=True)
    if os.listdir(repo_dir):
        try:
            existing = git_wrapper.get_repo_remote_url(repo_dir)
            existing = git_wrapper.remove_token_from_url(existing)
            if existing == url:
                return False
        except GitError:
            pass
    git_wrapper.clone(base_dir, token, url, repo_name, revision)
    return True


class Repository:
    """A local representation of a model Git repository on ModelScope Hub."""

    def __init__(self,
                 model_dir: str,
                 clone_from: str,
                 revision: Optional[str] = DEFAULT_REPOSITORY_REVISION,
                 auth_token: Optional[str] = None,
                 git_path: Optional[str] = None,
                 endpoint: Optional[str] = None):
        if not revision:
            raise InvalidParameter(
                'a non-default value of revision cannot be empty.')

        self._endpoint = endpoint
        self.model_dir = model_dir
        self.model_base_dir = os.path.dirname(model_dir)
        self.model_repo_name = os.path.basename(model_dir)
        self.auth_token = _resolve_token(auth_token)

        self.git_wrapper = GitCommandWrapper(git_path)
        if not self.git_wrapper.is_lfs_installed():
            logger.error('git lfs is not installed, please install.')

        url = self._get_model_id_url(clone_from)
        cloned = _clone_if_needed(
            self.git_wrapper, self.model_base_dir, self.model_repo_name,
            self.model_dir, url, self.auth_token, revision)
        if not cloned:
            return

        if self.git_wrapper.is_lfs_installed():
            self.git_wrapper.git_lfs_install(self.model_dir)

        self.git_wrapper.add_user_info(
            self.model_base_dir, self.model_repo_name)
        if self.auth_token:
            self.git_wrapper.config_auth_token(
                self.model_dir, self.auth_token)

    def _get_model_id_url(self, model_id: str) -> str:
        endpoint = self._endpoint or get_endpoint()
        return f'{endpoint}/{model_id}.git'

    def pull(self, remote: str = 'origin', branch: str = 'master'):
        """Pull *remote*/*branch* into the local checkout."""
        self.git_wrapper.pull(self.model_dir, remote=remote, branch=branch)

    def add_lfs_type(self, file_name_suffix: str) -> None:
        """Track an additional file-name pattern with Git LFS."""
        attrs = os.path.join(self.model_dir, '.gitattributes')
        with open(attrs, 'a', encoding='utf-8') as f:
            f.write(
                f'\n{file_name_suffix} filter=lfs diff=lfs merge=lfs -text\n')

    def push(self,
             commit_message: str,
             local_branch: Optional[str] = DEFAULT_REPOSITORY_REVISION,
             remote_branch: Optional[str] = DEFAULT_REPOSITORY_REVISION,
             force: bool = False):
        """Stage all changes, commit, and push to the remote."""
        warnings.warn(
            'This function is deprecated and will be removed in future '
            'versions. Please use git command directly or use '
            'HubApi().upload_folder instead',
            DeprecationWarning,
            stacklevel=2)
        if not isinstance(commit_message, str) or not commit_message:
            raise InvalidParameter('commit_message must be provided!')
        if not isinstance(force, bool):
            raise InvalidParameter('force must be bool')
        if not self.auth_token:
            raise NotLoginException('Must login to push, please login first.')

        self.git_wrapper.config_auth_token(self.model_dir, self.auth_token)
        self.git_wrapper.add_user_info(
            self.model_base_dir, self.model_repo_name)
        url = self.git_wrapper.get_repo_remote_url(self.model_dir)

        self.git_wrapper.add(self.model_dir, all_files=True)
        self.git_wrapper.commit(self.model_dir, commit_message)
        self.git_wrapper.push(
            repo_dir=self.model_dir,
            token=self.auth_token,
            url=url,
            local_branch=local_branch,
            remote_branch=remote_branch,
            force=force)

    def tag(self,
            tag_name: str,
            message: str,
            ref: Optional[str] = MASTER_MODEL_BRANCH):
        """Create an annotated tag pointing to *ref*."""
        if not tag_name:
            raise InvalidParameter(
                'We use tag-based revision, therefore tag_name '
                'cannot be None or empty.')
        if not message:
            raise InvalidParameter(
                'We use annotated tag, therefore message '
                'cannot None or empty.')
        self.git_wrapper.tag(
            repo_dir=self.model_dir,
            tag_name=tag_name,
            message=message,
            ref=ref)

    def tag_and_push(self,
                     tag_name: str,
                     message: str,
                     ref: Optional[str] = MASTER_MODEL_BRANCH):
        """Create *tag_name* and push it to the remote."""
        self.tag(tag_name, message, ref)
        self.git_wrapper.push_tag(
            repo_dir=self.model_dir, tag_name=tag_name)


class DatasetRepository:
    """A local representation of a dataset (metadata) Git repository."""

    def __init__(self,
                 repo_work_dir: str,
                 dataset_id: str,
                 revision: Optional[str] = DEFAULT_DATASET_REVISION,
                 auth_token: Optional[str] = None,
                 git_path: Optional[str] = None,
                 endpoint: Optional[str] = None):
        if not repo_work_dir or not isinstance(repo_work_dir, str):
            raise InvalidParameter('dataset_work_dir must be provided!')
        repo_work_dir = repo_work_dir.rstrip('/')
        if not repo_work_dir:
            raise InvalidParameter('dataset_work_dir can not be root dir!')
        if not revision:
            raise InvalidParameter(
                'a non-default value of revision cannot be empty.')

        self._endpoint = endpoint
        self.dataset_id = dataset_id
        self.repo_work_dir = repo_work_dir
        self.repo_base_dir = os.path.dirname(repo_work_dir)
        self.repo_name = os.path.basename(repo_work_dir)
        self.revision = revision
        self.auth_token = _resolve_token(auth_token)

        self.git_wrapper = GitCommandWrapper(git_path)
        os.makedirs(self.repo_work_dir, exist_ok=True)
        self.repo_url = self._get_repo_url(dataset_id)

    def _get_repo_url(self, dataset_id: str) -> str:
        endpoint = self._endpoint or get_endpoint()
        return f'{endpoint}/datasets/{dataset_id}.git'

    def clone(self) -> str:
        """Clone the dataset repo if not already cloned, returning its path."""
        cloned = _clone_if_needed(
            self.git_wrapper, self.repo_base_dir, self.repo_name,
            self.repo_work_dir, self.repo_url, self.auth_token, self.revision)
        return self.repo_work_dir if cloned else ''

    def push(self,
             commit_message: str,
             branch: Optional[str] = DEFAULT_DATASET_REVISION,
             force: bool = False):
        """Stage all changes, commit, and push to the remote."""
        warnings.warn(
            'This function is deprecated and will be removed in future '
            'versions. Please use git command directly or use '
            'HubApi().upload_folder instead',
            DeprecationWarning,
            stacklevel=2)
        if not isinstance(commit_message, str) or not commit_message:
            raise InvalidParameter('commit_message must be provided!')
        if not isinstance(force, bool):
            raise InvalidParameter('force must be bool')
        if not self.auth_token:
            raise NotLoginException('Must login to push, please login first.')

        self.git_wrapper.config_auth_token(self.repo_work_dir, self.auth_token)
        self.git_wrapper.add_user_info(self.repo_base_dir, self.repo_name)
        try:
            remote_url = self.git_wrapper.get_repo_remote_url(
                self.repo_work_dir)
            remote_url = self.git_wrapper.remove_token_from_url(remote_url)
        except GitError:
            remote_url = self.repo_url

        self.git_wrapper.pull(self.repo_work_dir)
        self.git_wrapper.add(self.repo_work_dir, all_files=True)
        self.git_wrapper.commit(self.repo_work_dir, commit_message)
        self.git_wrapper.push(
            repo_dir=self.repo_work_dir,
            token=self.auth_token,
            url=remote_url,
            local_branch=branch,
            remote_branch=branch,
            force=force)
