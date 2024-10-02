# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

from modelscope.cli.base import CLICommand
from modelscope.hub.constants import TEMPORARY_FOLDER_NAME


def subparser_func(args):
    """ Function which will be called for a specific sub parser.
    """
    return ClearCacheCMD(args)


class ClearCacheCMD(CLICommand):
    name = 'clear-cache'

    def __init__(self, args):
        self.args = args
        self.cache_dir = os.getenv(
            'MODELSCOPE_CACHE',
            Path.home().joinpath('.cache', 'modelscope'))

    @staticmethod
    def define_args(parsers: ArgumentParser):
        """ define args for clear-cache command.
        """
        parser = parsers.add_parser(ClearCacheCMD.name)
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            '--model',
            type=str,
            help=
            'The id of the model whose cache will be cleared. For clear-cache, '
            'if neither model or dataset id is provided, entire cache will be cleared.'
        )
        group.add_argument(
            '--dataset',
            type=str,
            help=
            'The id of the dataset whose cache will be cleared. For clear-cahe, '
            'if neither model or dataset id is provided, entire cache will be cleared.'
        )

        parser.set_defaults(func=subparser_func)

    def execute(self):
        self._execute_with_confirmation()

    def _execute_with_confirmation(self):
        all = False
        single_model = False
        prompt = '\nYou are about to delete '

        if self.args.model or self.args.dataset:
            if self.args.model:
                id = self.args.model
                single_model = True
                prompt = prompt + f'local cache for model {id}. '
            else:
                id = self.args.dataset
                prompt = prompt + f'local cache for dataset {id}. '
        else:
            prompt = prompt + f'entire ModelScope cache at {self.cache_dir}, including ALL models and dataset.\n'
            all = True
        user_input = input(
            prompt
            + '\nPlease press Y or y to proceed, any other key to abort.\n'
        ).strip().upper()

        if user_input == 'Y':
            if all:
                self._remove_directory(self.cache_dir)
                print('Cache cleared.')
            else:
                entity_directory = os.path.join(
                    self.cache_dir, 'hub' if single_model else 'datasets', id)
                temp_directory = os.path.join(
                    self.cache_dir, 'hub' if single_model else 'datasets',
                    TEMPORARY_FOLDER_NAME, id)
                entity_removed = self._remove_directory(entity_directory)
                temp_removed = self._remove_directory(temp_directory)
                if (not entity_removed) and (not temp_removed):
                    if single_model:
                        print(
                            f'Cache for Model {id} not found. Nothing to do.')
                    else:
                        print(
                            f'Cache for Dataset {id} not found. Nothing to do.'
                        )
                else:
                    print('Cache cleared.')
        else:
            print('Operation aborted.')
            return

    def _remove_directory(self, path):
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                print(f'Cache folder {path} removed.')
                return True
            except Exception as e:
                print(f'An error occurred while clearing cache at {path}: {e}')
            return False
