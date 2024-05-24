# Copyright (c) Alibaba, Inc. and its affiliates.

from argparse import ArgumentParser

from modelscope.cli.base import CLICommand
from modelscope.hub.file_download import model_file_download
from modelscope.hub.snapshot_download import snapshot_download


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
        parser.add_argument(
            '--model',
            type=str,
            required=True,
            help='The model id to be downloaded.')
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
        if len(self.args.files) == 1:  # download single file
            model_file_download(
                self.args.model,
                self.args.files[0],
                cache_dir=self.args.cache_dir,
                local_dir=self.args.local_dir,
                revision=self.args.revision)
        elif len(self.args.files) > 1:  # download specified multiple files.
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
