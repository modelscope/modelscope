# Copyright (c) Alibaba, Inc. and its affiliates.
"""ModelScope Studio runtime management CLI.

This module exposes a ``modelscope studio`` command group that wraps the
Studio OpenAPI methods on :class:`~modelscope.hub.api.HubApi`. Each CLI
subcommand is a thin adapter that parses arguments, delegates to a single
``HubApi`` method and renders a concise human-readable result.

Subcommand layout::

    modelscope studio deploy   <studio_id>
    modelscope studio stop     <studio_id>
    modelscope studio logs     <studio_id> [--type ...] [--keyword ...] ...
    modelscope studio settings <studio_id> [--display-name ...] ...
    modelscope studio secret   list   <studio_id>
    modelscope studio secret   add    <studio_id> <key> <value>
    modelscope studio secret   update <studio_id> <key> <value>
    modelscope studio secret   delete <studio_id> <key>
"""
from argparse import ArgumentParser, _SubParsersAction

import json

from modelscope.cli.base import CLICommand
from modelscope.hub.api import HubApi
from modelscope.hub.utils.utils import resolve_endpoint
from modelscope.utils.logger import get_logger

logger = get_logger()

# Common log type choices for `studio logs --type ...`.
_LOG_TYPES = ('run', 'build')


def subparser_func(args):
    """ Function which will be called for a specific sub parser.
    """
    return StudioCMD(args)


class StudioCMD(CLICommand):
    """Studio runtime management commands.

    Dispatches to a per-subcommand handler based on ``args.studio_action``
    (and, for ``secret``, ``args.secret_action``).
    """

    name = 'studio'

    def __init__(self, args):
        self.args = args

    # ------------------------------------------------------------------
    # Argument parsing
    # ------------------------------------------------------------------
    @staticmethod
    def define_args(parsers: _SubParsersAction):
        parser: ArgumentParser = parsers.add_parser(
            StudioCMD.name, help='Manage ModelScope studios at runtime.')
        # Common args available to every studio subcommand.
        parser.add_argument(
            '--token',
            type=str,
            default=None,
            help='Optional access token used for authentication.')
        parser.add_argument(
            '--endpoint',
            type=str,
            default=None,
            help='ModelScope server endpoint. Falls back to env '
            'MODELSCOPE_DOMAIN, then https://www.modelscope.cn.')

        action_subparsers = parser.add_subparsers(
            dest='studio_action', help='Studio subcommands.')

        StudioCMD._add_simple_id_parser(
            action_subparsers,
            name='deploy',
            help_text='Deploy (re-pull and rebuild) a studio.')
        StudioCMD._add_simple_id_parser(
            action_subparsers, name='stop', help_text='Stop a running studio.')
        StudioCMD._add_logs_parser(action_subparsers)
        StudioCMD._add_settings_parser(action_subparsers)
        StudioCMD._add_secret_parser(action_subparsers)

        parser.set_defaults(func=subparser_func)

    @staticmethod
    def _add_simple_id_parser(action_subparsers, name, help_text):
        sub = action_subparsers.add_parser(name, help=help_text)
        sub.add_argument(
            'studio_id',
            type=str,
            help='Studio ID in the format `owner/repo_name`.')

    @staticmethod
    def _add_logs_parser(action_subparsers):
        sub = action_subparsers.add_parser(
            'logs', help='Fetch studio runtime or build logs.')
        sub.add_argument(
            'studio_id',
            type=str,
            help='Studio ID in the format `owner/repo_name`.')
        sub.add_argument(
            '--type',
            dest='log_type',
            choices=_LOG_TYPES,
            default='run',
            help="Log type to fetch (defaults to 'run').")
        sub.add_argument(
            '--keyword',
            type=str,
            default=None,
            help='Optional keyword to filter log lines.')
        sub.add_argument(
            '--page-num',
            dest='page_num',
            type=int,
            default=1,
            help='Page number, starting from 1.')
        sub.add_argument(
            '--page-size',
            dest='page_size',
            type=int,
            default=100,
            help='Number of log entries per page.')
        sub.add_argument(
            '--start-timestamp',
            dest='start_timestamp',
            type=int,
            default=None,
            help='Optional start timestamp (seconds since epoch).')
        sub.add_argument(
            '--end-timestamp',
            dest='end_timestamp',
            type=int,
            default=None,
            help='Optional end timestamp (seconds since epoch).')

    @staticmethod
    def _add_settings_parser(action_subparsers):
        sub = action_subparsers.add_parser(
            'settings',
            help='Update studio settings (only specified fields are modified).'
        )
        sub.add_argument(
            'studio_id',
            type=str,
            help='Studio ID in the format `owner/repo_name`.')
        sub.add_argument(
            '--display-name',
            dest='display_name',
            type=str,
            default=None,
            help='Studio display name (Chinese name).')
        sub.add_argument(
            '--description',
            type=str,
            default=None,
            help='Studio description.')
        sub.add_argument(
            '--license', type=str, default=None, help='Studio license.')
        sub.add_argument(
            '--cover-image',
            dest='cover_image',
            type=str,
            default=None,
            help='Studio cover image URL.')
        sub.add_argument(
            '--sdk-type',
            dest='sdk_type',
            choices=['gradio', 'streamlit', 'docker', 'static'],
            default=None,
            help='Studio SDK type (requires redeployment).')
        sub.add_argument(
            '--sdk-version',
            dest='sdk_version',
            type=str,
            default=None,
            help='Studio SDK version (requires redeployment).')
        sub.add_argument(
            '--base-image',
            dest='base_image',
            type=str,
            default=None,
            help='Studio base image (requires redeployment).')
        sub.add_argument(
            '--hardware',
            type=str,
            default=None,
            help='Studio hardware configuration (requires redeployment).')
        visibility_group = sub.add_mutually_exclusive_group()
        visibility_group.add_argument(
            '--private',
            dest='private',
            action='store_const',
            const=True,
            default=None,
            help='Mark the studio as private.')
        visibility_group.add_argument(
            '--public',
            dest='private',
            action='store_const',
            const=False,
            help='Mark the studio as public.')

    @staticmethod
    def _add_secret_parser(action_subparsers):
        secret_parser = action_subparsers.add_parser(
            'secret', help='Manage studio environment variables (secrets).')
        secret_subparsers = secret_parser.add_subparsers(
            dest='secret_action', help='Secret subcommands.')

        list_sub = secret_subparsers.add_parser(
            'list', help='List secret keys (values are not returned).')
        list_sub.add_argument('studio_id', type=str)

        add_sub = secret_subparsers.add_parser('add', help='Add a secret.')
        add_sub.add_argument('studio_id', type=str)
        add_sub.add_argument('key', type=str, help='Secret name.')
        add_sub.add_argument('value', type=str, help='Secret value.')

        upd_sub = secret_subparsers.add_parser(
            'update', help='Update an existing secret.')
        upd_sub.add_argument('studio_id', type=str)
        upd_sub.add_argument('key', type=str, help='Secret name.')
        upd_sub.add_argument('value', type=str, help='New secret value.')

        del_sub = secret_subparsers.add_parser(
            'delete', help='Delete a secret.')
        del_sub.add_argument('studio_id', type=str)
        del_sub.add_argument('key', type=str, help='Secret name to delete.')

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def execute(self):
        action = getattr(self.args, 'studio_action', None)
        if action is None:
            raise SystemExit(
                'No studio subcommand specified. '
                "Run 'modelscope studio --help' to see available subcommands.")

        endpoint = resolve_endpoint(self.args.endpoint)
        api = HubApi(endpoint=endpoint)

        handlers = {
            'deploy': self._do_deploy,
            'stop': self._do_stop,
            'logs': self._do_logs,
            'settings': self._do_settings,
            'secret': self._do_secret,
        }
        handler = handlers.get(action)
        if handler is None:
            raise SystemExit(f'Unknown studio subcommand: {action}')
        handler(api, endpoint)

    # -- deploy / stop --------------------------------------------------
    def _do_deploy(self, api, endpoint):
        data = api.deploy_studio(
            self.args.studio_id, token=self.args.token, endpoint=endpoint)
        print(f'Deploy triggered for studio {self.args.studio_id}.')
        self._print_status(data)

    def _do_stop(self, api, endpoint):
        data = api.stop_studio(
            self.args.studio_id, token=self.args.token, endpoint=endpoint)
        print(f'Stop triggered for studio {self.args.studio_id}.')
        self._print_status(data)

    @staticmethod
    def _print_status(data):
        if not data:
            return
        status = data.get('status') if isinstance(data, dict) else None
        if status:
            print(f'Status: {status}')
        else:
            print(json.dumps(data, ensure_ascii=False, indent=2))

    # -- logs -----------------------------------------------------------
    def _do_logs(self, api, endpoint):
        data = api.get_studio_logs(
            self.args.studio_id,
            log_type=self.args.log_type,
            page_num=self.args.page_num,
            page_size=self.args.page_size,
            keyword=self.args.keyword,
            start_timestamp=self.args.start_timestamp,
            end_timestamp=self.args.end_timestamp,
            token=self.args.token,
            endpoint=endpoint)
        if not isinstance(data, dict):
            print(data)
            return
        # Server response shape may be {'logs': [...], 'total': N} or similar.
        logs = data.get('logs')
        if logs is None:
            # Fall back to dumping the raw payload so users can still see it.
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return
        for entry in logs:
            if isinstance(entry, dict):
                ts = entry.get('timestamp') or entry.get('time') or ''
                msg = entry.get('content') or entry.get('message') or ''
                line = f'[{ts}] {msg}' if ts else str(msg)
            else:
                line = str(entry)
            print(line)
        total = data.get('total')
        if total is not None:
            print(
                f'-- page {self.args.page_num} (size {self.args.page_size}), '
                f'total {total} --')

    # -- settings -------------------------------------------------------
    def _do_settings(self, api, endpoint):
        fields = [
            'display_name', 'description', 'license', 'cover_image',
            'sdk_type', 'sdk_version', 'base_image', 'hardware', 'private'
        ]
        settings = {
            f: getattr(self.args, f)
            for f in fields if getattr(self.args, f, None) is not None
        }
        if not settings:
            raise SystemExit(
                'No setting specified. Provide at least one of: '
                '--display-name, --description, --license, --cover-image, '
                '--sdk-type, --sdk-version, --base-image, --hardware, '
                '--private/--public.')
        data = api.update_studio_settings(
            self.args.studio_id,
            token=self.args.token,
            endpoint=endpoint,
            **settings)
        print(f'Updated settings for studio {self.args.studio_id}: '
              f"{', '.join(sorted(settings))}.")
        if data:
            print(json.dumps(data, ensure_ascii=False, indent=2))

    # -- secret ---------------------------------------------------------
    def _do_secret(self, api, endpoint):
        action = getattr(self.args, 'secret_action', None)
        if action is None:
            raise SystemExit(
                'No secret subcommand specified. '
                "Run 'modelscope studio secret --help' for usage.")
        if action == 'list':
            secrets = api.list_studio_secrets(
                self.args.studio_id, token=self.args.token, endpoint=endpoint)
            if not secrets:
                print('(no secrets)')
                return
            for item in secrets:
                key = item.get('key') if isinstance(item, dict) else item
                print(key)
        elif action == 'add':
            api.add_studio_secret(
                self.args.studio_id,
                self.args.key,
                self.args.value,
                token=self.args.token,
                endpoint=endpoint)
            print(f'Secret {self.args.key!r} added.')
        elif action == 'update':
            api.update_studio_secret(
                self.args.studio_id,
                self.args.key,
                self.args.value,
                token=self.args.token,
                endpoint=endpoint)
            print(f'Secret {self.args.key!r} updated.')
        elif action == 'delete':
            api.delete_studio_secret(
                self.args.studio_id,
                self.args.key,
                token=self.args.token,
                endpoint=endpoint)
            print(f'Secret {self.args.key!r} deleted.')
        else:
            raise SystemExit(f'Unknown secret subcommand: {action}')
