# Copyright (c) Alibaba, Inc. and its affiliates.
"""``modelscope server`` — launch the local inference HTTP server."""

import logging
from argparse import ArgumentParser

from modelscope_hub.cli.base import CLICommand

from modelscope.server.api_server import add_server_args, run_server
from modelscope.utils.logger import get_logger

logger = get_logger(log_level=logging.WARNING)


class ServerCMD(CLICommand):
    name = 'server'

    @staticmethod
    def register(subparsers: ArgumentParser) -> None:
        parser = subparsers.add_parser(
            ServerCMD.name, help='Launch the local inference HTTP server.')
        add_server_args(parser)
        parser.set_defaults(_command=ServerCMD)

    def execute(self):
        run_server(self.args)
