# Copyright (c) Alibaba, Inc. and its affiliates.

from argparse import ArgumentParser

from modelscope.cli.base import CLICommand
from modelscope.hub.file_download import (dataset_file_download,
                                          model_file_download)
from modelscope.hub.snapshot_download import (dataset_snapshot_download,
                                              snapshot_download)
from modelscope.utils.constant import DEFAULT_DATASET_REVISION


def subparser_func(args):
    """ Function which will be called for a specific sub parser.
    """
    return DownloadCMD(args)


class DownloadCMD(CLICommand):
    name = 'download'

    def __init__(self, args):
        self.args = args

    @staticmethod
    def define_args(parsers: ArgumentParser):
        """ define args for download command.
        """
        parser: ArgumentParser = parsers.add_parser(DownloadCMD.name)
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            '--model',
            type=str,
            help='The id of the model to be downloaded. For download, '
            'the id of either a model or dataset must be provided.')
        group.add_argument(
            '--dataset',
            type=str,
            help='The id of the dataset to be downloaded. For download, '
            'the id of either a model or dataset must be provided.')
        parser.add_argument(
            'repo_id',
            type=str,
            nargs='?',
            default=None,
            help='Optional, '
            'ID of the repo to download, It can also be set by --model or --dataset.'
        )
        parser.add_argument(
            '--repo-type',
            choices=['model', 'dataset'],
            default='model',
            help="Type of repo to download from (defaults to 'model').",
        )
        parser.add_argument(
            '--revision',
            type=str,
            default=None,
            help='Revision of the model.')
        parser.add_argument(
            '--cache_dir',
            type=str,
            default=None,
            help='Cache directory to save model.')
        parser.add_argument(
            '--local_dir',
            type=str,
            default=None,
            help='File will be downloaded to local location specified by'
            'local_dir, in this case, cache_dir parameter will be ignored.')
        parser.add_argument(
            'files',
            type=str,
            default=None,
            nargs='*',
            help='Specify relative path to the repository file(s) to download.'
            "(e.g 'tokenizer.json', 'onnx/decoder_model.onnx').")
        parser.add_argument(
            '--include',
            nargs='*',
            default=None,
            type=str,
            help='Glob patterns to match files to download.'
            'Ignored if file is specified')
        parser.add_argument(
            '--exclude',
            nargs='*',
            type=str,
            default=None,
            help='Glob patterns to exclude from files to download.'
            'Ignored if file is specified')
        parser.set_defaults(func=subparser_func)

    def execute(self):
        if self.args.model or self.args.dataset:
            # the position argument of files will be put to repo_id.
            if self.args.repo_id is not None:
                if self.args.files:
                    self.args.files.insert(0, self.args.repo_id)
                else:
                    self.args.files = [self.args.repo_id]
        else:
            if self.args.repo_id is not None:
                if self.args.repo_type == 'model':
                    self.args.model = self.args.repo_id
                elif self.args.repo_type == 'dataset':
                    self.args.dataset = self.args.repo_id
                else:
                    raise Exception('Not support repo-type: %s'
                                    % self.args.repo_type)
        if not self.args.model and not self.args.dataset:
            raise Exception('Model or dataset must be set.')
        if self.args.model:
            if len(self.args.files) == 1:  # download single file
                model_file_download(
                    self.args.model,
                    self.args.files[0],
                    cache_dir=self.args.cache_dir,
                    local_dir=self.args.local_dir,
                    revision=self.args.revision)
            elif len(
                    self.args.files) > 1:  # download specified multiple files.
                snapshot_download(
                    self.args.model,
                    revision=self.args.revision,
                    cache_dir=self.args.cache_dir,
                    local_dir=self.args.local_dir,
                    allow_file_pattern=self.args.files,
                )
            else:  # download repo
                snapshot_download(
                    self.args.model,
                    revision=self.args.revision,
                    cache_dir=self.args.cache_dir,
                    local_dir=self.args.local_dir,
                    allow_file_pattern=self.args.include,
                    ignore_file_pattern=self.args.exclude,
                )
        elif self.args.dataset:
            dataset_revision: str = self.args.revision if self.args.revision else DEFAULT_DATASET_REVISION
            if len(self.args.files) == 1:  # download single file
                dataset_file_download(
                    self.args.dataset,
                    self.args.files[0],
                    cache_dir=self.args.cache_dir,
                    local_dir=self.args.local_dir,
                    revision=dataset_revision)
            elif len(
                    self.args.files) > 1:  # download specified multiple files.
                dataset_snapshot_download(
                    self.args.dataset,
                    revision=dataset_revision,
                    cache_dir=self.args.cache_dir,
                    local_dir=self.args.local_dir,
                    allow_file_pattern=self.args.files,
                )
            else:  # download repo
                dataset_snapshot_download(
                    self.args.dataset,
                    revision=dataset_revision,
                    cache_dir=self.args.cache_dir,
                    local_dir=self.args.local_dir,
                    allow_file_pattern=self.args.include,
                    ignore_file_pattern=self.args.exclude,
                )
        else:
            pass  # noop
