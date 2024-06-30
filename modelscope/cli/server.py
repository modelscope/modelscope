# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os
from argparse import ArgumentParser
from string import Template

from modelscope.cli.base import CLICommand
from modelscope.server.api_server import add_server_args, run_server
from modelscope.utils.logger import get_logger

logger = get_logger(log_level=logging.WARNING)

current_path = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(current_path, 'template')


def subparser_func(args):
    """ Function which will be called for a specific sub parser.
    """
    return ServerCMD(args)


class ServerCMD(CLICommand):
    name = 'server'

    def __init__(self, args):
        self.args = args

    @staticmethod
    def define_args(parsers: ArgumentParser):
        """ define args for create pipeline template command.
        """
        parser = parsers.add_parser(ServerCMD.name)
        add_server_args(parser)
        parser.set_defaults(func=subparser_func)

    def execute(self):
        run_server(self.args)
