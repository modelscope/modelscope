# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from argparse import ArgumentParser, _SubParsersAction

from modelscope.cli.base import CLICommand
from modelscope.hub.api import HubApi, ModelScopeConfig
from modelscope.utils.constant import REPO_TYPE_MODEL, REPO_TYPE_SUPPORT
from modelscope.utils.logger import get_logger

logger = get_logger()


def subparser_func(args):
    """ Function which will be called for a specific sub parser.
    """
    return UploadCMD(args)


class UploadCMD(CLICommand):

    name = 'upload'

    def __init__(self, args: _SubParsersAction):
        self.args = args

    @staticmethod
    def define_args(parsers: _SubParsersAction):

        parser: ArgumentParser = parsers.add_parser(UploadCMD.name)

        parser.add_argument(
            'repo_id',
            type=str,
            help='The ID of the repo to upload to (e.g. `username/repo-name`)')
        parser.add_argument(
            'local_path',
            type=str,
            nargs='?',
            default=None,
            help='Optional, '
            'Local path to the file or folder to upload. Defaults to current directory.'
        )
        parser.add_argument(
            'path_in_repo',
            type=str,
            nargs='?',
            default=None,
            help='Optional, '
            'Path of the file or folder in the repo. Defaults to the relative path of the file or folder.'
        )
        parser.add_argument(
            '--repo-type',
            choices=REPO_TYPE_SUPPORT,
            default=REPO_TYPE_MODEL,
            help=
            'Type of the repo to upload to (e.g. `dataset`, `model`). Defaults to be `model`.',
        )
        parser.add_argument(
            '--include',
            nargs='*',
            type=str,
            help='Glob patterns to match files to upload.')
        parser.add_argument(
            '--exclude',
            nargs='*',
            type=str,
            help='Glob patterns to exclude from files to upload.')
        parser.add_argument(
            '--commit-message',
            type=str,
            default=None,
            help='The message of commit. Default to be `None`.')
        parser.add_argument(
            '--commit-description',
            type=str,
            default=None,
            help=
            'The description of the generated commit. Default to be `None`.')
        parser.add_argument(
            '--token',
            type=str,
            default=None,
            help=
            'A User Access Token generated from https://modelscope.cn/my/myaccesstoken'
        )
        parser.add_argument(
            '--max-workers',
            type=int,
            default=min(8,
                        os.cpu_count() + 4),
            help='The number of workers to use for uploading files.')
        parser.add_argument(
            '--endpoint',
            type=str,
            default='https://www.modelscope.cn',
            help='Endpoint for Modelscope service.')

        parser.set_defaults(func=subparser_func)

    def execute(self):

        assert self.args.repo_id, '`repo_id` is required'
        assert self.args.repo_id.count(
            '/') == 1, 'repo_id should be in format of username/repo-name'
        repo_name: str = self.args.repo_id.split('/')[-1]
        self.repo_id = self.args.repo_id

        # Check path_in_repo
        if self.args.local_path is None and os.path.isfile(repo_name):
            # Case 1: modelscope upload owner_name/test_repo
            self.local_path = repo_name
            self.path_in_repo = repo_name
        elif self.args.local_path is None and os.path.isdir(repo_name):
            # Case 2: modelscope upload owner_name/test_repo  (run command line in the `repo_name` dir)
            # => upload all files in current directory to remote root path
            self.local_path = repo_name
            self.path_in_repo = '.'
        elif self.args.local_path is None:
            # Case 3: user provided only a repo_id that does not match a local file or folder
            # => the user must explicitly provide a local_path => raise exception
            raise ValueError(
                f"'{repo_name}' is not a local file or folder. Please set `local_path` explicitly."
            )
        elif self.args.path_in_repo is None and os.path.isfile(
                self.args.local_path):
            # Case 4: modelscope upload owner_name/test_repo /path/to/your_file.csv
            # => upload it to remote root path with same name
            self.local_path = self.args.local_path
            self.path_in_repo = os.path.basename(self.args.local_path)
        elif self.args.path_in_repo is None:
            # Case 5: modelscope upload owner_name/test_repo /path/to/your_folder
            # => upload all files in current directory to remote root path
            self.local_path = self.args.local_path
            self.path_in_repo = ''
        else:
            # Finally, if both paths are explicit
            self.local_path = self.args.local_path
            self.path_in_repo = self.args.path_in_repo

        # Check token and login
        # The cookies will be reused if the user has logged in before.
        api = HubApi(endpoint=self.args.endpoint)

        if self.args.token:
            api.login(access_token=self.args.token)
        cookies = ModelScopeConfig.get_cookies()
        if cookies is None:
            raise ValueError(
                'The `token` is not provided! '
                'You can pass the `--token` argument, '
                'or use api.login(access_token=`your_sdk_token`). '
                'Your token is available at https://modelscope.cn/my/myaccesstoken'
            )

        if os.path.isfile(self.local_path):
            api.upload_file(
                path_or_fileobj=self.local_path,
                path_in_repo=self.path_in_repo,
                repo_id=self.repo_id,
                repo_type=self.args.repo_type,
                commit_message=self.args.commit_message,
                commit_description=self.args.commit_description,
            )
        elif os.path.isdir(self.local_path):
            api.upload_folder(
                repo_id=self.repo_id,
                folder_path=self.local_path,
                path_in_repo=self.path_in_repo,
                commit_message=self.args.commit_message,
                commit_description=self.args.commit_description,
                repo_type=self.args.repo_type,
                allow_patterns=self.args.include,
                ignore_patterns=self.args.exclude,
                max_workers=self.args.max_workers,
            )
        else:
            raise ValueError(f'{self.local_path} is not a valid local path')

        logger.info(f'Finished uploading to {self.repo_id}')
