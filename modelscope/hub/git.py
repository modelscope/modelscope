# Copyright (c) Alibaba, Inc. and its affiliates.
"""Git wrapper — shim delegating to ``modelscope_hub._git``.

Preserves the legacy ``GitCommandWrapper`` Singleton interface used
throughout the SDK while routing primitive Git operations through
:class:`modelscope_hub._git.GitCommand`.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional

from modelscope_hub._git import GitCommand as _GitCommand

from modelscope.hub.errors import GitError
from modelscope.utils.constant import MASTER_MODEL_BRANCH
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['GitError', 'GitCommandWrapper', 'Singleton']


class Singleton(type):
    """Metaclass enforcing one instance per class — preserved for parity."""

    _instances: dict = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class GitCommandWrapper(metaclass=Singleton):
    """Backward-compatible Git wrapper.

    Wraps :class:`modelscope_hub._git.GitCommand` to expose the legacy
    instance-method API (``clone``, ``push``, ``pull``, ``tag``…) used
    by callers that pre-date the SDK refactor.
    """

    default_git_path = 'git'

    def __init__(self, path: Optional[str] = None):
        self.git_path = path or self.default_git_path
        _GitCommand.set_git_path(self.git_path)

    # ------------------------------------------------------------------
    # Low-level subprocess passthrough (legacy contract)
    # ------------------------------------------------------------------
    def _run_git_command(self, *args):
        """Run a git subcommand, raising :class:`GitError` on failure."""
        try:
            return _GitCommand._run(*[a for a in args if a])
        except Exception as exc:  # _git.GitError → legacy GitError
            raise GitError(str(exc)) from exc

    # ------------------------------------------------------------------
    # URL / token helpers
    # ------------------------------------------------------------------
    def _add_git_token(self, git_token: str, url: str) -> str:
        return _GitCommand._inject_token(url, git_token)

    def remove_token_from_url(self, url: str) -> str:
        return _GitCommand.strip_token_from_url(url)

    # ------------------------------------------------------------------
    # LFS
    # ------------------------------------------------------------------
    def is_lfs_installed(self) -> bool:
        return _GitCommand.is_lfs_available()

    def git_lfs_install(self, repo_dir: str) -> bool:
        try:
            _GitCommand.lfs_install(Path(repo_dir))
            return True
        except Exception:
            return False

    def list_lfs_files(self, repo_dir: str) -> List[str]:
        rsp = self._run_git_command('-C', repo_dir, 'lfs', 'ls-files')
        return [
            line.split(' ')[-1] for line in rsp.stdout.strip().splitlines()
            if line
        ]

    # ------------------------------------------------------------------
    # Auth / user config
    # ------------------------------------------------------------------
    def config_git_token(self, repo_dir: str, git_token: str) -> None:
        url = self.get_repo_remote_url(repo_dir)
        if '//oauth2' in url:
            return
        auth_url = self._add_git_token(git_token, url)
        self._run_git_command('-C', repo_dir, 'remote', 'set-url', 'origin',
                              auth_url)

    def add_user_info(self, repo_base_dir: str, repo_name: str) -> None:
        from modelscope.hub.api import ModelScopeConfig
        user_name, user_email = ModelScopeConfig.get_user_info()
        if not (user_name and user_email):
            return
        repo_dir = os.path.join(repo_base_dir, repo_name)
        self._run_git_command('-C', repo_dir, 'config', 'user.name', user_name)
        self._run_git_command('-C', repo_dir, 'config', 'user.email',
                              user_email)

    # ------------------------------------------------------------------
    # Clone / pull / push
    # ------------------------------------------------------------------
    def clone(self,
              repo_base_dir: str,
              git_token: Optional[str],
              url: str,
              repo_name: str,
              branch: Optional[str] = None):
        target = Path(repo_base_dir) / repo_name
        try:
            _GitCommand.clone(
                url=url, target_dir=target, branch=branch, token=git_token)
        except Exception as exc:
            if (target / '.git').is_dir():
                logger.warning(
                    'git clone exited non-zero but repository was cloned '
                    'at %s. Likely a post-clone hook. Continuing.', target)
                return None
            raise GitError(str(exc)) from exc

    def pull(self,
             repo_dir: str,
             remote: str = 'origin',
             branch: str = 'master'):
        return self._run_git_command('-C', repo_dir, 'pull', remote, branch)

    def push(self,
             repo_dir: str,
             git_token: str,
             url: str,
             local_branch: str,
             remote_branch: str,
             force: bool = False):
        auth_url = self._add_git_token(git_token, url)
        args = [
            '-C', repo_dir, 'push', auth_url, f'{local_branch}:{remote_branch}'
        ]
        if force:
            args.append('-f')
        return self._run_git_command(*args)

    # ------------------------------------------------------------------
    # Add / commit / branch / checkout
    # ------------------------------------------------------------------
    def add(self,
            repo_dir: str,
            files: Optional[List[str]] = None,
            all_files: bool = False):
        if all_files:
            return self._run_git_command('-C', repo_dir, 'add', '-A')
        return self._run_git_command('-C', repo_dir, 'add', *(files or []))

    def commit(self, repo_dir: str, message: str):
        return self._run_git_command('-C', repo_dir, 'commit', '-m',
                                     f"'{message}'")

    def checkout(self, repo_dir: str, revision: str):
        return self._run_git_command('-C', repo_dir, 'checkout', revision)

    def new_branch(self, repo_dir: str, revision: str):
        return self._run_git_command('-C', repo_dir, 'checkout', '-b',
                                     revision)

    def get_remote_branches(self, repo_dir: str) -> List[str]:
        rsp = self._run_git_command('-C', repo_dir, 'branch', '-r')
        info = [
            line.strip() for line in rsp.stdout.strip().splitlines() if line
        ]
        if len(info) <= 1:
            return ['/'.join(info[0].split('/')[1:])] if info else []
        return ['/'.join(line.split('/')[1:]) for line in info[1:]]

    def get_repo_remote_url(self, repo_dir: str) -> str:
        rsp = self._run_git_command('-C', repo_dir, 'config', '--get',
                                    'remote.origin.url')
        return rsp.stdout.strip()

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------
    def tag(self,
            repo_dir: str,
            tag_name: str,
            message: str,
            ref: str = MASTER_MODEL_BRANCH):
        return self._run_git_command('-C', repo_dir, 'tag', tag_name, '-m',
                                     f'"{message}"', ref)

    def push_tag(self, repo_dir: str, tag_name: str):
        return self._run_git_command('-C', repo_dir, 'push', 'origin',
                                     tag_name)
