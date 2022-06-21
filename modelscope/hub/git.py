from threading import local
from tkinter.messagebox import NO
from typing import Union

from modelscope.utils.logger import get_logger
from .constants import LOGGER_NAME
from .utils._subprocess import run_subprocess

logger = get_logger


def git_clone(
    local_dir: str,
    repo_url: str,
):
    # TODO: use "git clone" or "git lfs clone" according to git version
    # TODO: print stderr when subprocess fails
    run_subprocess(
        f'git clone {repo_url}'.split(),
        local_dir,
        True,
    )


def git_checkout(
    local_dir: str,
    revsion: str,
):
    run_subprocess(f'git checkout {revsion}'.split(), local_dir)


def git_add(local_dir: str, ):
    run_subprocess(
        'git add .'.split(),
        local_dir,
        True,
    )


def git_commit(local_dir: str, commit_message: str):
    run_subprocess(
        'git commit -v -m'.split() + [commit_message],
        local_dir,
        True,
    )


def git_push(local_dir: str, branch: str):
    # check current branch
    cur_branch = git_current_branch(local_dir)
    if cur_branch != branch:
        logger.error(
            "You're trying to push to a different branch, please double check")
        return

    run_subprocess(
        f'git push origin {branch}'.split(),
        local_dir,
        True,
    )


def git_current_branch(local_dir: str) -> Union[str, None]:
    """
    Get current branch name

    Args:
        local_dir(`str`): local model repo directory

    Returns
        branch name you're currently on
    """
    try:
        process = run_subprocess(
            'git rev-parse --abbrev-ref HEAD'.split(),
            local_dir,
            True,
        )

        return str(process.stdout).strip()
    except Exception as e:
        raise e
