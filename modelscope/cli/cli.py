# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import logging

from modelscope.cli.clearcache import ClearCacheCMD
from modelscope.cli.download import DownloadCMD
from modelscope.cli.llamafile import LlamafileCMD
from modelscope.cli.login import LoginCMD
from modelscope.cli.modelcard import ModelCardCMD
from modelscope.cli.pipeline import PipelineCMD
from modelscope.cli.plugins import PluginsCMD
from modelscope.cli.server import ServerCMD
from modelscope.hub.api import HubApi
from modelscope.utils.logger import get_logger

logger = get_logger(log_level=logging.WARNING)


def run_cmd():
    parser = argparse.ArgumentParser(
        'ModelScope Command Line tool', usage='modelscope <command> [<args>]')
    parser.add_argument(
        '--token', default=None, help='Specify ModelScope SDK token.')
    subparsers = parser.add_subparsers(help='modelscope commands helpers')

    DownloadCMD.define_args(subparsers)
    ClearCacheCMD.define_args(subparsers)
    PluginsCMD.define_args(subparsers)
    PipelineCMD.define_args(subparsers)
    ModelCardCMD.define_args(subparsers)
    ServerCMD.define_args(subparsers)
    LoginCMD.define_args(subparsers)
    LlamafileCMD.define_args(subparsers)

    args = parser.parse_args()

    if not hasattr(args, 'func'):
        parser.print_help()
        exit(1)
    if args.token is not None:
        api = HubApi()
        api.login(args.token)
    cmd = args.func(args)
    cmd.execute()


if __name__ == '__main__':
    run_cmd()
