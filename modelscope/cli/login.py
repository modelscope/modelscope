# Copyright (c) Alibaba, Inc. and its affiliates.

from argparse import ArgumentParser

from modelscope.cli.base import CLICommand
from modelscope.hub.api import HubApi
from modelscope.hub.utils.utils import resolve_endpoint


def subparser_func(args):
    """ Function which will be called for a specific sub parser.
    """
    return LoginCMD(args)


class LoginCMD(CLICommand):
    name = 'login'

    def __init__(self, args):
        self.args = args

    @staticmethod
    def define_args(parsers: ArgumentParser):
        """ define args for login command.
        """
        parser = parsers.add_parser(LoginCMD.name)
        parser.add_argument(
            '--token',
            type=str,
            required=True,
            help='The Access Token for modelscope.')
        parser.add_argument(
            '--endpoint',
            type=str,
            default=None,
            help='ModelScope server endpoint, e.g. modelscope.cn or '
            'https://modelscope.cn. Scheme (https://) is auto-completed '
            'if omitted. Falls back to env MODELSCOPE_DOMAIN, then '
            'defaults to https://www.modelscope.cn.')
        parser.set_defaults(func=subparser_func)

    def execute(self):
        api = HubApi(endpoint=resolve_endpoint(self.args.endpoint))
        api.login(self.args.token)
