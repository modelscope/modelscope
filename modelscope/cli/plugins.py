# Copyright (c) Alibaba, Inc. and its affiliates.
"""``modelscope plugin`` — install/uninstall/list ModelScope plugins."""

from argparse import ArgumentParser

from modelscope_hub.cli.base import CLICommand

from modelscope.utils.plugins import PluginsManager

plugins_manager = PluginsManager()


class PluginsCMD(CLICommand):
    name = 'plugin'

    @staticmethod
    def register(subparsers: ArgumentParser) -> None:
        parser = subparsers.add_parser(
            PluginsCMD.name, help='Manage ModelScope plugins.')
        sub = parser.add_subparsers(dest='command')

        install = sub.add_parser('install', help='Install plugin packages.')
        install.add_argument(
            'package',
            type=str,
            nargs='+',
            default=None,
            help='Name of the package to be installed.')
        install.add_argument(
            '--index_url',
            '-i',
            type=str,
            default=None,
            help='Base URL of the Python Package Index.')
        install.add_argument(
            '--force_update',
            '-f',
            type=str,
            default=False,
            help='If force update the package')

        uninstall = sub.add_parser(
            'uninstall', help='Uninstall plugin packages.')
        uninstall.add_argument(
            'package',
            type=str,
            nargs='+',
            default=None,
            help='Name of the package to be uninstalled.')
        uninstall.add_argument(
            '--yes',
            '-y',
            type=str,
            default=False,
            help='Skip confirmation prompt.')

        list_p = sub.add_parser('list', help='List available plugins.')
        list_p.add_argument(
            '--all',
            '-a',
            type=str,
            default=None,
            help='Show all of the plugins including those not installed.')

        parser.set_defaults(_command=PluginsCMD)

    def execute(self):
        command = getattr(self.args, 'command', None)
        if command == 'install':
            plugins_manager.install_plugins(
                list(self.args.package),
                index_url=self.args.index_url,
                force_update=self.args.force_update)
        elif command == 'uninstall':
            plugins_manager.uninstall_plugins(
                list(self.args.package), is_yes=self.args.yes)
        elif command == 'list':
            plugins_manager.list_plugins(show_all=self.args.all)
        else:
            raise ValueError(
                'Usage: modelscope plugin {install|uninstall|list} ...')
