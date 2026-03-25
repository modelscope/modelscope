# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os
import sys
from argparse import ArgumentParser

from modelscope.cli.base import CLICommand
from modelscope.hub.api import HubApi
from modelscope.hub.constants import DEFAULT_MAX_WORKERS
from modelscope.hub.file_download import (dataset_file_download,
                                          model_file_download)
from modelscope.hub.snapshot_download import (dataset_snapshot_download,
                                              snapshot_download)
from modelscope.hub.utils.utils import convert_patterns
from modelscope.utils.constant import DEFAULT_DATASET_REVISION
from modelscope.utils.logger import get_logger

logger = get_logger(log_level=logging.WARNING)


def subparser_func(args):
    """ Function which will be called for a specific sub parser.
    """
    return DownloadCMD(args)


class DownloadCMD(CLICommand):
    name = 'download'

    def __init__(self, args):
        self.args = args

    @staticmethod
    def define_args(parsers: ArgumentParser):
        """ define args for download command.
        """
        parser: ArgumentParser = parsers.add_parser(DownloadCMD.name)
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            '--model',
            type=str,
            help='The id of the model to be downloaded. For download, '
            'the id of either a model or dataset must be provided.')
        group.add_argument(
            '--dataset',
            type=str,
            help='The id of the dataset to be downloaded. For download, '
            'the id of either a model or dataset must be provided.')
        group.add_argument(
            '--collection',
            type=str,
            default=None,
            help='The ID of the collection to download (skills only)')
        parser.add_argument(
            'repo_id',
            type=str,
            nargs='?',
            default=None,
            help='Optional, '
            'ID of the repo to download, It can also be set by --model or --dataset.'
        )
        parser.add_argument(
            '--repo-type',
            choices=['model', 'dataset'],
            default='model',
            help="Type of repo to download from (defaults to 'model').",
        )
        parser.add_argument(
            '--token',
            type=str,
            default=None,
            help='Optional. Access token to download controlled entities.')
        parser.add_argument(
            '--revision',
            type=str,
            default=None,
            help='Revision of the entity (e.g., model).')
        parser.add_argument(
            '--cache_dir',
            type=str,
            default=None,
            help='Cache directory to save entity (e.g., model).')
        parser.add_argument(
            '--local_dir',
            type=str,
            default=None,
            help='File will be downloaded to local location specified by'
            'local_dir, in this case, cache_dir parameter will be ignored.')
        parser.add_argument(
            'files',
            type=str,
            default=None,
            nargs='*',
            help='Specify relative path to the repository file(s) to download.'
            "(e.g 'tokenizer.json', 'onnx/decoder_model.onnx').")
        parser.add_argument(
            '--include',
            nargs='*',
            default=None,
            type=str,
            help='Glob patterns to match files to download.'
            'Ignored if file is specified')
        parser.add_argument(
            '--exclude',
            nargs='*',
            type=str,
            default=None,
            help='Glob patterns to exclude from files to download.'
            'Ignored if file is specified')
        parser.add_argument(
            '--max-workers',
            type=int,
            default=DEFAULT_MAX_WORKERS,
            help='The maximum number of workers to download files.')

        parser.set_defaults(func=subparser_func)

    def execute(self):
        if self.args.model or self.args.dataset:
            # the position argument of files will be put to repo_id.
            if self.args.repo_id is not None:
                if self.args.files:
                    self.args.files.insert(0, self.args.repo_id)
                else:
                    self.args.files = [self.args.repo_id]
        else:
            if self.args.repo_id is not None:
                if self.args.repo_type == 'model':
                    self.args.model = self.args.repo_id
                elif self.args.repo_type == 'dataset':
                    self.args.dataset = self.args.repo_id
                else:
                    raise Exception('Not support repo-type: %s'
                                    % self.args.repo_type)
        if not self.args.model and not self.args.dataset and not self.args.collection:
            raise Exception('Model, dataset, or collection must be set.')
        cookies = None
        if self.args.token is not None:
            api = HubApi()
            cookies = api.get_cookies(access_token=self.args.token)
        if self.args.model:
            if len(self.args.files) == 1:  # download single file
                model_file_download(
                    self.args.model,
                    self.args.files[0],
                    cache_dir=self.args.cache_dir,
                    local_dir=self.args.local_dir,
                    revision=self.args.revision,
                    cookies=cookies)
            elif len(
                    self.args.files) > 1:  # download specified multiple files.
                snapshot_download(
                    self.args.model,
                    revision=self.args.revision,
                    cache_dir=self.args.cache_dir,
                    local_dir=self.args.local_dir,
                    allow_file_pattern=self.args.files,
                    max_workers=self.args.max_workers,
                    cookies=cookies)
            else:  # download repo
                snapshot_download(
                    self.args.model,
                    revision=self.args.revision,
                    cache_dir=self.args.cache_dir,
                    local_dir=self.args.local_dir,
                    allow_file_pattern=convert_patterns(self.args.include),
                    ignore_file_pattern=convert_patterns(self.args.exclude),
                    max_workers=self.args.max_workers,
                    cookies=cookies)
            print(f'\nSuccessfully Downloaded from model {self.args.model}.\n')
        elif self.args.dataset:
            dataset_revision: str = self.args.revision if self.args.revision else DEFAULT_DATASET_REVISION
            if len(self.args.files) == 1:  # download single file
                dataset_file_download(
                    self.args.dataset,
                    self.args.files[0],
                    cache_dir=self.args.cache_dir,
                    local_dir=self.args.local_dir,
                    revision=dataset_revision,
                    cookies=cookies)
            elif len(
                    self.args.files) > 1:  # download specified multiple files.
                dataset_snapshot_download(
                    self.args.dataset,
                    revision=dataset_revision,
                    cache_dir=self.args.cache_dir,
                    local_dir=self.args.local_dir,
                    allow_file_pattern=self.args.files,
                    max_workers=self.args.max_workers,
                    cookies=cookies)
            else:  # download repo
                dataset_snapshot_download(
                    self.args.dataset,
                    revision=dataset_revision,
                    cache_dir=self.args.cache_dir,
                    local_dir=self.args.local_dir,
                    allow_file_pattern=convert_patterns(self.args.include),
                    ignore_file_pattern=convert_patterns(self.args.exclude),
                    max_workers=self.args.max_workers,
                    cookies=cookies)
            print(
                f'\nSuccessfully Downloaded from dataset {self.args.dataset}.\n'
            )
        elif self.args.collection:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            api = HubApi(token=self.args.token)
            local_dir = self.args.local_dir or os.path.join(
                os.path.expanduser('~'), '.agents', 'skills')
            data = api.get_collection(self.args.collection, repo_type='skill')
            elements = data.get('CollectionElements',
                                {}).get('CollectionElementVoList', [])

            logger.info(
                f'Collection {self.args.collection} has {len(elements)} elements.'
            )

            if not elements:
                print('No skill elements found in collection: %s'
                      % self.args.collection)
                return

            # Validate elements have required fields
            valid_elements = []
            for elem in elements:
                if not elem.get('ElementPath') or not elem.get('ElementName'):
                    logger.warning('Skipping malformed collection element: %s',
                                   elem)
                    continue
                valid_elements.append(elem)

            if not valid_elements:
                print('No valid skill elements found in collection: %s'
                      % self.args.collection)
                return

            print('Found %d skill(s) in collection, downloading...'
                  % len(valid_elements))

            succeeded = []
            failed = []

            def _download_one_skill(element):
                element_path = element['ElementPath']
                element_name = element['ElementName']
                skill_id = '%s/%s' % (element_path, element_name)
                try:
                    skill_dir = api.download_skill(
                        skill_id=skill_id, local_dir=local_dir)
                    return (element_path, element_name, skill_dir, None)
                except Exception as e:
                    return (element_path, element_name, None, str(e))

            with ThreadPoolExecutor(
                    max_workers=self.args.max_workers) as executor:
                futures = {
                    executor.submit(_download_one_skill, elem): elem
                    for elem in valid_elements
                }
                for future in as_completed(futures):
                    path, name, skill_dir, error = future.result()
                    if error:
                        failed.append((path, name, error))
                        print('Failed to download skill %s/%s: %s' %
                              (path, name, error))
                    else:
                        succeeded.append((path, name, skill_dir))
                        print('Downloaded skill %s/%s -> %s' %
                              (path, name, skill_dir))

            print('\nDownload complete: %d succeeded, %d failed' %
                  (len(succeeded), len(failed)))
            if failed:
                print('Failed skills:')
                for path, name, error in failed:
                    print('  %s/%s: %s' % (path, name, error))
                sys.exit(1)
        else:
            pass  # noop
