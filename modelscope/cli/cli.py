# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import logging

from modelscope.cli.clearcache import ClearCacheCMD
from modelscope.cli.create import CreateCMD
from modelscope.cli.download import DownloadCMD
from modelscope.cli.llamafile import LlamafileCMD
from modelscope.cli.login import LoginCMD
from modelscope.cli.modelcard import ModelCardCMD
from modelscope.cli.pipeline import PipelineCMD
from modelscope.cli.plugins import PluginsCMD
from modelscope.cli.scancache import ScanCacheCMD
from modelscope.cli.server import ServerCMD
from modelscope.cli.upload import UploadCMD
from modelscope.hub.constants import MODELSCOPE_ASCII
from modelscope.utils.logger import get_logger
from modelscope.version import __version__

logger = get_logger(log_level=logging.WARNING)


def run_cmd():
    print(MODELSCOPE_ASCII)
    parser = argparse.ArgumentParser(
        'ModelScope Command Line tool', usage='modelscope <command> [<args>]')
    parser.add_argument(
        '-V',
        '--version',
        action='version',
        version=f'ModelScope CLI {__version__}')
    parser.add_argument(
        '--token', default=None, help='Specify ModelScope SDK token.')
    subparsers = parser.add_subparsers(help='modelscope commands helpers')

    CreateCMD.define_args(subparsers)
    DownloadCMD.define_args(subparsers)
    UploadCMD.define_args(subparsers)
    ClearCacheCMD.define_args(subparsers)
    PluginsCMD.define_args(subparsers)
    PipelineCMD.define_args(subparsers)
    ModelCardCMD.define_args(subparsers)
    ServerCMD.define_args(subparsers)
    LoginCMD.define_args(subparsers)
    LlamafileCMD.define_args(subparsers)
    ScanCacheCMD.define_args(subparsers)

    args = parser.parse_args()

    if not hasattr(args, 'func'):
        parser.print_help()
        exit(1)
    cmd = args.func(args)
    cmd.execute()


if __name__ == '__main__':
    run_cmd()
