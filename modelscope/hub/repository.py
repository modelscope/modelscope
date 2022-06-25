import os
import subprocess
from pathlib import Path
from typing import Optional, Union

from modelscope.utils.logger import get_logger
from .api import ModelScopeConfig
from .constants import MODELSCOPE_URL_SCHEME
from .git import git_add, git_checkout, git_clone, git_commit, git_push
from .utils._subprocess import run_subprocess
from .utils.utils import get_gitlab_domain

logger = get_logger()


class Repository:

    def __init__(
        self,
        local_dir: str,
        clone_from: Optional[str] = None,
        auth_token: Optional[str] = None,
        private: Optional[bool] = False,
        revision: Optional[str] = 'master',
    ):
        """
        Instantiate a Repository object by cloning the remote ModelScopeHub repo
        Args:
            local_dir(`str`):
                local directory to store the model files
            clone_from(`Optional[str] = None`):
                model id in ModelScope-hub from which git clone
                You should ignore this parameter when `local_dir` is already a git repo
            auth_token(`Optional[str]`):
                token obtained when calling `HubApi.login()`. Usually you can safely ignore the parameter
                as the token is already saved when you login the first time
            private(`Optional[bool]`):
                whether the model is private, default to False
            revision(`Optional[str]`):
                revision of the model you want to clone from. Can be any of a branch, tag or commit hash
        """
        logger.info('Instantiating Repository object...')

        # Create local directory if not exist
        os.makedirs(local_dir, exist_ok=True)
        self.local_dir = os.path.join(os.getcwd(), local_dir)

        self.private = private

        # Check git and git-lfs installation
        self.check_git_versions()

        # Retrieve auth token
        if not private and isinstance(auth_token, str):
            logger.warning(
                'cloning a public repo with a token, which will be ignored')
            self.token = None
        else:
            if isinstance(auth_token, str):
                self.token = auth_token
            else:
                self.token = ModelScopeConfig.get_token()

            if self.token is None:
                raise EnvironmentError(
                    'Token does not exist, the clone will fail for private repo.'
                    'Please login first.')

        # git clone
        if clone_from is not None:
            self.model_id = clone_from
            logger.info('cloning model repo to %s ...', self.local_dir)
            git_clone(self.local_dir, self.get_repo_url())
        else:
            if is_git_repo(self.local_dir):
                logger.debug('[Repository] is a valid git repo')
            else:
                raise ValueError(
                    'If not specifying `clone_from`, you need to pass Repository a'
                    ' valid git clone.')

        # git checkout
        if isinstance(revision, str) and revision != 'master':
            git_checkout(revision)

    def push_to_hub(self,
                    commit_message: str,
                    revision: Optional[str] = 'master'):
        """
        Push changes changes to hub

        Args:
            commit_message(`str`):
                commit message describing the changes, it's mandatory
            revision(`Optional[str]`):
                remote branch you want to push to, default to `master`

        <Tip>
            The function complains when local and remote branch are different, please be careful
        </Tip>

        """
        git_add(self.local_dir)
        git_commit(self.local_dir, commit_message)

        logger.info('Pushing changes to repo...')
        git_push(self.local_dir, revision)

        # TODO: if git push fails, how to retry?

    def check_git_versions(self):
        """
        Checks that `git` and `git-lfs` can be run.

        Raises:
            `EnvironmentError`: if `git` or `git-lfs` are not installed.
        """
        try:
            git_version = run_subprocess('git --version'.split(),
                                         self.local_dir).stdout.strip()
        except FileNotFoundError:
            raise EnvironmentError(
                'Looks like you do not have git installed, please install.')

        try:
            lfs_version = run_subprocess('git-lfs --version'.split(),
                                         self.local_dir).stdout.strip()
        except FileNotFoundError:
            raise EnvironmentError(
                'Looks like you do not have git-lfs installed, please install.'
                ' You can install from https://git-lfs.github.com/.'
                ' Then run `git lfs install` (you only have to do this once).')
        logger.info(git_version + '\n' + lfs_version)

    def get_repo_url(self) -> str:
        """
        Get repo url to clone, according whether the repo is private or not
        """
        url = None

        if self.private:
            url = f'{MODELSCOPE_URL_SCHEME}oauth2:{self.token}@{get_gitlab_domain()}/{self.model_id}'
        else:
            url = f'{MODELSCOPE_URL_SCHEME}{get_gitlab_domain()}/{self.model_id}'

        if not url:
            raise ValueError(
                'Empty repo url, please check clone_from parameter')

        logger.debug('url to clone: %s', str(url))

        return url


def is_git_repo(folder: Union[str, Path]) -> bool:
    """
    Check if the folder is the root or part of a git repository

    Args:
        folder (`str`):
            The folder in which to run the command.

    Returns:
        `bool`: `True` if the repository is part of a repository, `False`
        otherwise.
    """
    folder_exists = os.path.exists(os.path.join(folder, '.git'))
    git_branch = subprocess.run(
        'git branch'.split(),
        cwd=folder,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    return folder_exists and git_branch.returncode == 0
