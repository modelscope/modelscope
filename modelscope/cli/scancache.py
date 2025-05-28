# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os
import time
from argparse import ArgumentParser
from typing import Optional

from modelscope.cli.base import CLICommand
from modelscope.hub.cache_manager import scan_cache_dir
from modelscope.hub.errors import CacheNotFound
from modelscope.utils.logger import get_logger

logger = get_logger(log_level=logging.WARNING)

current_path = os.path.dirname(os.path.abspath(__file__))


def subparser_func(args):
    """ Function which will be called for a specific sub parser.
    """
    return ScanCacheCMD(args)


class ScanCacheCMD(CLICommand):
    name = 'scan-cache'

    def __init__(self, args):
        self.args = args
        self.cache_dir: Optional[str] = args.dir

    @staticmethod
    def define_args(parsers: ArgumentParser):
        """ define args for create pipeline template command.
        """
        parser = parsers.add_parser(ScanCacheCMD.name)
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            '--dir',
            type=str,
            default=None,
            help=
            'cache directory to scan (optional). Default to the default ModelScope cache.',
        )

        parser.set_defaults(func=subparser_func)

    def execute(self):
        try:
            t0 = time.time()
            cache_info = scan_cache_dir(self.cache_dir)
            t1 = time.time()
        except CacheNotFound as exc:
            cache_dir = exc.cache_dir
            print(f'Cache directory not found: {cache_dir}')
            return
        print(cache_info.export_as_table())
        print(
            f'\nDone in {round(t1 - t0, 1)}s. Scanned {len(cache_info.repos)} repo(s)'
            f' for a total of {cache_info.size_on_disk_str}.')
        if len(cache_info.warnings) > 0:
            message = f'Got {len(cache_info.warnings)} warning(s) while scanning.'
            print(message)
            for warning in cache_info.warnings:
                print(warning)
