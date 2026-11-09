# Copyright (c) Alibaba, Inc. and its affiliates.
"""Clear-cache CLI command — retained for backward compatibility.

The ``modelscope_hub`` CLI now owns ``clear-cache`` as an alias for
``cache clear``, but this module preserves the legacy :class:`ClearCacheCMD`
class so that existing tests and callers that import it directly continue
to work.
"""
import os
from argparse import ArgumentParser
from pathlib import Path

from modelscope_hub import get_cache_root
from modelscope_hub._cache_manager import (_cache_clear_targets, _safe_rmtree,
                                           clear_cache)

from modelscope.cli.base import CLICommand


def subparser_func(args):
    """ Function which will be called for a specific sub parser.
    """
    return ClearCacheCMD(args)


class ClearCacheCMD(CLICommand):
    name = 'clear-cache'

    def __init__(self, args):
        self.args = args
        self.cache_dir = str(get_cache_root())

    @staticmethod
    def register(subparsers) -> None:
        """Register clear-cache subcommand (CLICommand ABC contract)."""
        ClearCacheCMD.define_args(subparsers)

    @staticmethod
    def define_args(parsers: ArgumentParser):
        """ define args for clear-cache command.
        """
        parser = parsers.add_parser(ClearCacheCMD.name)
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            '--model',
            type=str,
            help='The id of the model whose cache will be cleared. '
            'If neither model or dataset id is provided, entire cache '
            'will be cleared.')
        group.add_argument(
            '--dataset',
            type=str,
            help='The id of the dataset whose cache will be cleared. '
            'If neither model or dataset id is provided, entire cache '
            'will be cleared.')

        parser.set_defaults(func=subparser_func)

    def execute(self):
        self._execute_with_confirmation()

    def _execute_with_confirmation(self):
        model = getattr(self.args, 'model', None)
        dataset = getattr(self.args, 'dataset', None)
        repo_type = 'model' if model is not None else (
            'dataset' if dataset is not None else None)
        repo_id = model if model is not None else dataset
        root, targets = _cache_clear_targets(self.cache_dir, repo_type,
                                             repo_id)
        print(f'Cache root: {root}')
        for target in targets:
            print(f'  {target}')
        if not targets:
            print('Cache not found. Nothing to do.')
            return
        user_input = input(
            '\nPlease press Y or y to clear the listed repositories, any other key to abort.\n'
        ).strip().upper()
        if user_input != 'Y':
            print('Operation aborted.')
            return
        clear_cache(cache_dir=root, repo_type=repo_type, repo_id=repo_id)
        print('Cache cleared.')

    def _remove_directory(self, path):
        """Remove a directory within the cache root without following symlinks."""
        if os.path.lexists(path):
            _safe_rmtree(
                Path(path),
                Path(self.cache_dir).expanduser().resolve())
            print(f'Cache folder {path} removed.')
            return True
        return False
