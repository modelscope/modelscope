# Copyright (c) Alibaba, Inc. and its affiliates.
"""``modelscope llamafile`` — download and run a llamafile from a model repo."""

import logging
import os
import sys
from argparse import ArgumentParser

from modelscope_hub.cli.base import CLICommand

from modelscope import model_file_download
from modelscope.hub.api import HubApi
from modelscope.utils.logger import get_logger

logger = get_logger(log_level=logging.WARNING)


class LlamafileCMD(CLICommand):
    name = 'llamafile'

    def __init__(self, args):
        super().__init__(args)
        self.model_id = self.args.model
        if self.model_id is None or self.model_id.count('/') != 1:
            raise ValueError(f'Invalid model id [{self.model_id}].')
        if self.args.file is not None:
            # ignore accuracy if file argument is provided
            self.args.accuracy = None
            if not self.args.file.lower().endswith('.llamafile'):
                raise ValueError('file argument must ends with ".llamafile".')
        self.api = HubApi()

    @staticmethod
    def register(subparsers: ArgumentParser) -> None:
        parser = subparsers.add_parser(
            LlamafileCMD.name,
            help='Download and run a llamafile from a model repo.')
        parser.add_argument(
            '--model',
            type=str,
            required=True,
            help=
            'The id of the model, whose repo must contain at least one llamafile'
        )
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            '--accuracy',
            type=str,
            required=False,
            default='q4_k_m',
            help=
            'Selected accuracy of GGUF files in the repo. Ignored when "file" is also provided.'
        )

        group.add_argument(
            '--file',
            type=str,
            required=False,
            help=
            'The name of a specified llamafile in the model repo. This takes precedence over "accuracy".'
        )

        parser.add_argument(
            '--local_dir',
            type=str,
            default=None,
            help=
            'Directory where the selected llamafile would will be downloaded to.'
        )

        group.add_argument(
            '--launch',
            type=str,
            required=False,
            default='True',
            help=
            'Whether to launch model with the downloaded llamafile, default to True.'
        )

        parser.set_defaults(_command=LlamafileCMD)

    def execute(self):
        if self.args.file:
            self.args.accuracy = None

        all_files = self.api.get_model_files(self.model_id, recursive=True)
        llamafiles = []
        for info in all_files:
            file_path = info['Path']
            if file_path and file_path.lower().endswith(
                    '.llamafile') and '-of-' not in file_path.lower():
                llamafiles.append(file_path)
        if not llamafiles:
            raise ValueError(
                f'Cannot locate a valid llamafile in repo {self.model_id}.')
        logger.info(
            f'list of llamafiles in repo {self.model_id}:\n{llamafiles}.')
        # default choose the first llamafile if there is no q4_k_m, and no accuracy or file is specified
        selected_file = llamafiles[0]
        found = False
        for f in llamafiles:
            if self.args.file and f == self.args.file:
                selected_file = f
                found = True
                break
            if self.args.accuracy and self.args.accuracy.lower() in f.lower():
                selected_file = f
                found = True
                break
        if found:
            print(f'llamafile matching criteria found: [{selected_file}].')
        else:
            print(
                f'No matched llamafile found in repo, choosing the first llamafile in repo: [{selected_file}]'
            )
        downloaded_file = os.path.abspath(
            model_file_download(
                self.args.model, selected_file, local_dir=self.args.local_dir))

        if sys.platform.startswith('win'):
            downloaded_file = self._rename_extension(downloaded_file)

        if self.args.launch.lower() == 'true':
            print(f'Launching model with llamafile [{downloaded_file}]:')
            self._execute_llamafile(downloaded_file)
        else:
            print(
                f'No Launching. Llamafile model downloaded to [{downloaded_file}], you may execute it separately.'
            )

    def _execute_llamafile(self, file_path):
        current_mode = os.stat(file_path).st_mode
        new_mode = current_mode | 0o111
        os.chmod(file_path, new_mode)
        execute_cmd = file_path
        has_gpu = False
        try:
            import torch
            has_gpu = torch.cuda.is_available()
        except ModuleNotFoundError:
            # we depend on torch to detect gpu.
            # if torch is not available, we will just assume gpu cannot be used
            pass
        if has_gpu:
            print(
                'GPU detected, launching model with llamafile GPU option >>>')
            execute_cmd = f'{execute_cmd} -ngl 999'
        os.system(execute_cmd)

    def _rename_extension(self, original_file_name):
        directory, filename = os.path.split(original_file_name)
        base_name, _ = os.path.splitext(filename)
        new_filename = os.path.join(directory, f'{base_name}.exe')
        os.rename(original_file_name, new_filename)
        return new_filename
