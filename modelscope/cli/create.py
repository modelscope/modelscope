# Copyright (c) Alibaba, Inc. and its affiliates.
from argparse import ArgumentParser, _SubParsersAction

from modelscope.cli.base import CLICommand
from modelscope.hub.api import HubApi
from modelscope.hub.constants import (Licenses, ModelVisibility, Visibility,
                                      VisibilityMap)
from modelscope.hub.utils.aigc import AigcModel
from modelscope.utils.constant import REPO_TYPE_MODEL, REPO_TYPE_SUPPORT
from modelscope.utils.logger import get_logger

logger = get_logger()


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

        # AIGC specific arguments
        aigc_group = parser.add_argument_group(
            'AIGC Model Creation',
            'Arguments for creating an AIGC model. Use --aigc to enable.')
        aigc_group.add_argument(
            '--aigc', action='store_true', help='Enable AIGC model creation.')
        aigc_group.add_argument(
            '--from_json',
            type=str,
            help='Path to a JSON file containing AIGC model configuration. '
            'If used, all other parameters except --repo_id are ignored.')
        aigc_group.add_argument(
            '--model_path', type=str, help='Path to the model file or folder.')
        aigc_group.add_argument(
            '--aigc_type',
            type=str,
            help="AIGC type. Recommended: 'Checkpoint', 'LoRA', 'VAE'.")
        aigc_group.add_argument(
            '--base_model_type',
            type=str,
            help='Base model type, e.g., SD_XL.')
        aigc_group.add_argument(
            '--revision',
            type=str,
            default='v1.0',
            help="Model revision. Defaults to 'v1.0'.")
        aigc_group.add_argument(
            '--base_model_id',
            type=str,
            default='',
            help='Base model ID from ModelScope.')
        aigc_group.add_argument(
            '--description',
            type=str,
            default='This is an AIGC model.',
            help='Model description.')
        aigc_group.add_argument(
            '--path_in_repo',
            type=str,
            default='',
            help='Path in the repository to upload to.')
        aigc_group.add_argument(
            '--model_source',
            type=str,
            default='USER_UPLOAD',
            help=
            'Source of the AIGC model. `USER_UPLOAD`, `TRAINED_FROM_MODELSCOPE` or `TRAINED_FROM_ALIYUN_FC`.'
        )

        parser.set_defaults(func=subparser_func)

    def execute(self):
        if self.args.aigc:
            if self.args.repo_type != REPO_TYPE_MODEL:
                raise ValueError(
                    'AIGC models can only be created when repo_type is "model".'
                )
            self._create_aigc_model()
        else:
            self._create_regular_repo()

    def _create_regular_repo(self):
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

    def _create_aigc_model(self):
        """Execute the command."""
        api = HubApi(endpoint=self.args.endpoint)
        model_id = self.args.repo_id

        if self.args.from_json:
            # Create from JSON file
            logger.info('Creating AIGC model from JSON file: '
                        f'{self.args.from_json}')
            aigc_model = AigcModel.from_json_file(self.args.from_json)
        else:
            # Create from command line arguments
            logger.info('Creating AIGC model from command line arguments...')
            if not all([
                    self.args.model_path, self.args.aigc_type,
                    self.args.base_model_type
            ]):
                raise ValueError(
                    'Error: --model_path, --aigc_type, and '
                    '--base_model_type are required when not using '
                    '--from_json.')

            aigc_model = AigcModel(
                model_path=self.args.model_path,
                aigc_type=self.args.aigc_type,
                base_model_type=self.args.base_model_type,
                tag=self.args.revision,
                description=self.args.description,
                base_model_id=self.args.base_model_id,
                path_in_repo=self.args.path_in_repo,
                model_source=self.args.model_source,
            )

        # Convert visibility string to int for the API call
        reverse_visibility_map = {v: k for k, v in VisibilityMap.items()}
        visibility_idx: int = reverse_visibility_map.get(
            self.args.visibility, ModelVisibility.PUBLIC)

        try:
            model_url = api.create_model(
                model_id=model_id,
                token=self.args.token,
                visibility=visibility_idx,
                license=self.args.license,
                chinese_name=self.args.chinese_name,
                aigc_model=aigc_model)
            print(f'Successfully created AIGC model: {model_url}')
        except Exception as e:
            print(f'Error creating AIGC model: {e}')
