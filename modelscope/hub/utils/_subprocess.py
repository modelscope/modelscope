import subprocess
from typing import List


def run_subprocess(command: List[str],
                   folder: str,
                   check=True,
                   **kwargs) -> subprocess.CompletedProcess:
    """
    Method to run subprocesses. Calling this will capture the `stderr` and `stdout`,
    please call `subprocess.run` manually in case you would like for them not to
    be captured.

    Args:
        command (`List[str]`):
            The command to execute as a list of strings.
        folder (`str`):
            The folder in which to run the command.
        check (`bool`, *optional*, defaults to `True`):
            Setting `check` to `True` will raise a `subprocess.CalledProcessError`
            when the subprocess has a non-zero exit code.
        kwargs (`Dict[str]`):
            Keyword arguments to be passed to the `subprocess.run` underlying command.

    Returns:
        `subprocess.CompletedProcess`: The completed process.
    """
    if isinstance(command, str):
        raise ValueError(
            '`run_subprocess` should be called with a list of strings.')

    return subprocess.run(
        command,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        check=check,
        encoding='utf-8',
        cwd=folder,
        **kwargs,
    )
