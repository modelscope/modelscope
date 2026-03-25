# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import sys
from argparse import ArgumentParser

from modelscope.cli.base import CLICommand
from modelscope.cli.utils import concurrent_download
from modelscope.hub.api import HubApi
from modelscope.hub.constants import DEFAULT_SKILLS_DIR
from modelscope.utils.logger import get_logger

logger = get_logger(log_level=logging.WARNING)


def subparser_func(args):
    """Function which will be called for a specific sub parser."""
    return SkillsCMD(args)


class SkillsCMD(CLICommand):
    """Command for managing skills."""

    name = 'skills'

    def __init__(self, args):
        self.args = args

    @staticmethod
    def define_args(parsers: ArgumentParser):
        """Define args for skills command."""
        parser = parsers.add_parser(SkillsCMD.name)
        subparsers = parser.add_subparsers(
            dest='skills_action', help='skills subcommands')

        # 'add' subcommand
        add_parser = subparsers.add_parser(
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
        add_parser.set_defaults(func=subparser_func)

    def execute(self):
        if not hasattr(self.args,
                       'skills_action') or not self.args.skills_action:
            print('Usage: modelscope skills add <skill_id1> <skill_id2> ...')
            return

        if not hasattr(self.args, 'skill_ids') or not self.args.skill_ids:
            print('No skill IDs provided. Usage: modelscope skills add '
                  '<skill_id1> <skill_id2> ...')
            return

        api = HubApi(token=self.args.token)
        local_dir = self.args.local_dir or DEFAULT_SKILLS_DIR

        skill_ids = self.args.skill_ids
        print(f'Downloading {len(skill_ids)} skill(s)...')

        if len(skill_ids) == 1:
            # Single skill download
            try:
                skill_dir = api.download_skill(
                    skill_id=skill_ids[0], local_dir=local_dir)
                print(f'Skill downloaded to: {skill_dir}')
            except Exception as e:
                print(f'Failed to download skill {skill_ids[0]}: {e}')
                sys.exit(1)
        else:
            # Multiple skills - concurrent download
            def _download_one(skill_id):
                try:
                    skill_dir = api.download_skill(
                        skill_id=skill_id, local_dir=local_dir)
                    return (skill_id, skill_dir, None)
                except Exception as e:
                    return (skill_id, None, str(e))

            concurrent_download(
                _download_one,
                skill_ids,
                max_workers=self.args.max_workers,
                item_name='skill')
