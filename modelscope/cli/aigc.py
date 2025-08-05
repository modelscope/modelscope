# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse

from modelscope.cli.base import CLICommand
from modelscope.hub.api import HubApi
from modelscope.hub.utils.aigc import AigcModel


class CreateAigcModelCMD(CLICommand):
    name = 'create-aigc-model'

    def __init__(self, args):
        self.args = args

    @staticmethod
    def define_args(parsers: argparse._SubParsersAction):
        """Define the arguments for the command."""
        parser = parsers.add_parser(
            CreateAigcModelCMD.name,
            description=
            'Create an AIGC model on ModelScope Hub from parameters or a JSON file.'
        )

        # Group for loading from JSON
        json_group = parser.add_argument_group(
            'From JSON', 'Create model from a JSON config file.')
        json_group.add_argument(
            '--from-json',
            type=str,
            help='Path to a JSON file containing AIGC model configuration. '
            'If used, all other parameters except --model-id are ignored.')

        # Group for creating from direct parameters
        param_group = parser.add_argument_group(
            'From Parameters', 'Create model by providing direct parameters.')
        param_group.add_argument(
            '--model-id',
            type=str,
            help='The model ID, e.g., your-namespace/your-model-name.')
        param_group.add_argument(
            '--model-path', type=str, help='Path to the model file or folder.')
        param_group.add_argument(
            '--aigc-type',
            type=str,
            help="AIGC type. Recommended: 'Checkpoint', 'LoRA', 'VAE'.")
        param_group.add_argument(
            '--base-model-type',
            type=str,
            help='Base model type, e.g., SD_XL.')
        param_group.add_argument(
            '--revision',
            type=str,
            default='v1.0',
            help="Model revision. Defaults to 'v1.0'.")
        param_group.add_argument(
            '--description',
            type=str,
            default='This is an AIGC model.',
            help='Model description.')
        param_group.add_argument(
            '--base-model-id',
            type=str,
            default='',
            help='Base model ID from ModelScope.')
        param_group.add_argument(
            '--path-in-repo',
            type=str,
            default='',
            help='Path in the repository to upload to.')
        parser.set_defaults(func=CreateAigcModelCMD)

    def execute(self):
        """Execute the command."""
        # Basic validation
        if not self.args.from_json and not self.args.model_id:
            print('Error: Either --from-json or --model-id must be provided.')
            return

        api = HubApi()
        api.login(self.args.token)

        if self.args.from_json:
            # Create from JSON file
            print('Creating AIGC model from JSON file: '
                  f'{self.args.from_json}')
            aigc_model = AigcModel.from_json_file(self.args.from_json)
            # model_id must still be provided if not in json, or for override
            model_id = self.args.model_id or aigc_model.model_id
            if not model_id:
                print("Error: --model-id is required when it's not present "
                      'in the JSON file.')
                return
        else:
            # Create from command line arguments
            print('Creating AIGC model from command line arguments...')
            model_id = self.args.model_id
            if not all([
                    self.args.model_path, self.args.aigc_type,
                    self.args.base_model_type
            ]):
                print('Error: --model-path, --aigc-type, and '
                      '--base-model-type are required when not using '
                      '--from-json.')
                return

            aigc_model = AigcModel(
                model_path=self.args.model_path,
                aigc_type=self.args.aigc_type,
                base_model_type=self.args.base_model_type,
                revision=self.args.revision,
                description=self.args.description,
                base_model_id=self.args.base_model_id,
                path_in_repo=self.args.path_in_repo,
            )

        try:
            model_url = api.create_model(
                model_id=model_id, aigc_model=aigc_model)
            print(f'Successfully created AIGC model: {model_url}')
        except Exception as e:
            print(f'Error creating AIGC model: {e}')
