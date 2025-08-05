# Copyright (c) Alibaba, Inc. and its affiliates.
from argparse import ArgumentParser, _SubParsersAction

from modelscope.cli.base import CLICommand
from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, Visibility
from modelscope.utils.constant import REPO_TYPE_MODEL, REPO_TYPE_SUPPORT


def subparser_func(args):
    """ Function which will be called for a specific sub parser.
    """
    return CreateCMD(args)


class CreateCMD(CLICommand):
    """
    Command for creating a new repository, supporting both model and dataset.
    """

    name = 'create'

    def __init__(self, args: _SubParsersAction):
        self.args = args

    @staticmethod
    def define_args(parsers: _SubParsersAction):

        parser: ArgumentParser = parsers.add_parser(CreateCMD.name)

        parser.add_argument(
            'repo_id',
            type=str,
            help='The ID of the repo to create (e.g. `username/repo-name`)')
        parser.add_argument(
            '--token',
            type=str,
            default=None,
            help=
            'A User Access Token generated from https://modelscope.cn/my/myaccesstoken to authenticate the user. '
            'If not provided, the CLI will use the local credentials if available.'
        )
        parser.add_argument(
            '--repo_type',
            choices=REPO_TYPE_SUPPORT,
            default=REPO_TYPE_MODEL,
            help=
            'Type of the repo to create (e.g. `dataset`, `model`). Default to `model`.',
        )
        parser.add_argument(
            '--visibility',
            choices=[
                Visibility.PUBLIC, Visibility.INTERNAL, Visibility.PRIVATE
            ],
            default=Visibility.PUBLIC,
            help='Visibility of the repo to create. Default to `public`.',
        )
        parser.add_argument(
            '--chinese_name',
            type=str,
            default=None,
            help='Optional, Chinese name of the repo. Default to `None`.',
        )
        parser.add_argument(
            '--license',
            type=str,
            choices=Licenses.to_list(),
            default=Licenses.APACHE_V2,
            help=
            'Optional, License of the repo. Default to `Apache License 2.0`.',
        )
        parser.add_argument(
            '--endpoint',
            type=str,
            default=None,
            help='Optional, The modelscope server address. Default to None.',
        )

        parser.set_defaults(func=subparser_func)

    def execute(self):

        # Check token and login
        # The cookies will be reused if the user has logged in before.
        api = HubApi(endpoint=self.args.endpoint)

        # Create repo
        api.create_repo(
            repo_id=self.args.repo_id,
            token=self.args.token,
            visibility=self.args.visibility,
            repo_type=self.args.repo_type,
            chinese_name=self.args.chinese_name,
            license=self.args.license,
            exist_ok=True,
            create_default_config=True,
            endpoint=self.args.endpoint,
        )

        print(f'Successfully created the repo: {self.args.repo_id}.')
