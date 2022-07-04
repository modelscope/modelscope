import os
from typing import List, Optional

from modelscope.hub.errors import GitError, InvalidParameter
from modelscope.utils.logger import get_logger
from .api import ModelScopeConfig
from .constants import MODELSCOPE_URL_SCHEME
from .git import GitCommandWrapper
from .utils.utils import get_endpoint

logger = get_logger()


class Repository:
    """Representation local model git repository.
    """

    def __init__(
        self,
        model_dir: str,
        clone_from: str,
        revision: Optional[str] = 'master',
        auth_token: Optional[str] = None,
        git_path: Optional[str] = None,
    ):
        """
        Instantiate a Repository object by cloning the remote ModelScopeHub repo
        Args:
            model_dir(`str`):
                The model root directory.
            clone_from:
                model id in ModelScope-hub from which git clone
            revision(`Optional[str]`):
                revision of the model you want to clone from. Can be any of a branch, tag or commit hash
            auth_token(`Optional[str]`):
                token obtained when calling `HubApi.login()`. Usually you can safely ignore the parameter
                as the token is already saved when you login the first time, if None, we will use saved token.
            git_path:(`Optional[str]`):
                The git command line path, if None, we use 'git'
        """
        self.model_dir = model_dir
        self.model_base_dir = os.path.dirname(model_dir)
        self.model_repo_name = os.path.basename(model_dir)
        if auth_token:
            self.auth_token = auth_token
        else:
            self.auth_token = ModelScopeConfig.get_token()

        git_wrapper = GitCommandWrapper()
        if not git_wrapper.is_lfs_installed():
            logger.error('git lfs is not installed, please install.')

        self.git_wrapper = GitCommandWrapper(git_path)
        os.makedirs(self.model_dir, exist_ok=True)
        url = self._get_model_id_url(clone_from)
        if os.listdir(self.model_dir):  # directory not empty.
            remote_url = self._get_remote_url()
            remote_url = self.git_wrapper.remove_token_from_url(remote_url)
            if remote_url and remote_url == url:  # need not clone again
                return
        self.git_wrapper.clone(self.model_base_dir, self.auth_token, url,
                               self.model_repo_name, revision)

        if git_wrapper.is_lfs_installed():
            git_wrapper.git_lfs_install(self.model_dir)  # init repo lfs

    def _get_model_id_url(self, model_id):
        url = f'{get_endpoint()}/{model_id}.git'
        return url

    def _get_remote_url(self):
        try:
            remote = self.git_wrapper.get_repo_remote_url(self.model_dir)
        except GitError:
            remote = None
        return remote

    def push(self,
             commit_message: str,
             branch: Optional[str] = 'master',
             force: bool = False):
        """Push local to remote, this method will do.
           git add
           git commit
           git push
        Args:
            commit_message (str): commit message
            revision (Optional[str], optional): which branch to push. Defaults to 'master'.
        """
        if commit_message is None or not isinstance(commit_message, str):
            msg = 'commit_message must be provided!'
            raise InvalidParameter(msg)
        if not isinstance(force, bool):
            raise InvalidParameter('force must be bool')
        url = self.git_wrapper.get_repo_remote_url(self.model_dir)
        self.git_wrapper.pull(self.model_dir)
        self.git_wrapper.add(self.model_dir, all_files=True)
        self.git_wrapper.commit(self.model_dir, commit_message)
        self.git_wrapper.push(
            repo_dir=self.model_dir,
            token=self.auth_token,
            url=url,
            local_branch=branch,
            remote_branch=branch)
