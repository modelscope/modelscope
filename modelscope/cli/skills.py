# Copyright (c) Alibaba, Inc. and its affiliates.
"""``modelscope skills`` — download and install agent skills."""

import logging
import sys
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

from modelscope_hub.cli.base import CLICommand

from modelscope.hub.api import HubApi
from modelscope.hub.constants import DEFAULT_SKILLS_DIR
from modelscope.utils.logger import get_logger

logger = get_logger(log_level=logging.WARNING)


def _concurrent_download(download_fn, items, max_workers=8, item_name='item'):
    """Run ``download_fn`` over ``items`` in parallel, reporting progress.

    ``download_fn`` must return ``(identifier, result_path, error_or_None)``.
    On any failure the process exits with status 1 after the summary is
    printed.
    """
    succeeded, failed = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_fn, item): item for item in items}
        for future in as_completed(futures):
            identifier, result_path, error = future.result()
            if error:
                failed.append((identifier, error))
                print(f'Failed to download {item_name} {identifier}: {error}')
            else:
                succeeded.append((identifier, result_path))
                print(f'Downloaded {item_name} {identifier} -> {result_path}')

    print(f'\nDownload complete: {len(succeeded)} succeeded, '
          f'{len(failed)} failed')
    if failed:
        print(f'Failed {item_name}s:')
        for identifier, error in failed:
            print(f'  {identifier}: {error}')
        sys.exit(1)
    return succeeded, failed


class SkillsCMD(CLICommand):
    """Command for managing skills."""

    name = 'skills'

    @staticmethod
    def register(subparsers: ArgumentParser) -> None:
        parser = subparsers.add_parser(
            SkillsCMD.name, help='Download and manage agent skills.')
        sub = parser.add_subparsers(
            dest='skills_action', help='skills subcommands')

        add_parser = sub.add_parser(
            'add', help='Download and install skills')
        add_parser.add_argument(
            'skill_ids',
            type=str,
            nargs='+',
            help='Skill IDs to download, in format: <path>/<name>')
        add_parser.add_argument(
            '--token',
            type=str,
            default=None,
            help='Access token for authentication')
        add_parser.add_argument(
            '--local_dir',
            type=str,
            default=None,
            help='Target directory for skills (default: ~/.agents/skills)')
        add_parser.add_argument(
            '--max-workers',
            type=int,
            default=8,
            help='Maximum concurrent downloads (default: 8)')

        parser.set_defaults(_command=SkillsCMD)

    def execute(self):
        if not getattr(self.args, 'skills_action', None):
            print('Usage: modelscope skills add <skill_id1> <skill_id2> ...')
            return

        skill_ids = getattr(self.args, 'skill_ids', None)
        if not skill_ids:
            print('No skill IDs provided. Usage: modelscope skills add '
                  '<skill_id1> <skill_id2> ...')
            return

        api = HubApi(token=self.args.token)
        local_dir = self.args.local_dir or DEFAULT_SKILLS_DIR

        print(f'Downloading {len(skill_ids)} skill(s)...')

        if len(skill_ids) == 1:
            try:
                skill_dir = api.download_skill(
                    skill_id=skill_ids[0], local_dir=local_dir)
                print(f'Skill downloaded to: {skill_dir}')
            except Exception as e:
                print(f'Failed to download skill {skill_ids[0]}: {e}')
                sys.exit(1)
        else:
            def _download_one(skill_id):
                try:
                    skill_dir = api.download_skill(
                        skill_id=skill_id, local_dir=local_dir)
                    return (skill_id, skill_dir, None)
                except Exception as e:
                    return (skill_id, None, str(e))

            _concurrent_download(
                _download_one,
                skill_ids,
                max_workers=self.args.max_workers,
                item_name='skill')
