# Copyright (c) Alibaba, Inc. and its affiliates.

from argparse import ArgumentParser

from modelscope.cli.base import CLICommand
from modelscope.utils.plugins import PluginsManager

plugins_manager = PluginsManager()


def subparser_func(args):
    """ Function which will be called for a specific sub parser.
    """
    return PluginsCMD(args)


class PluginsCMD(CLICommand):
    name = 'plugin'

    def __init__(self, args):
        self.args = args

    @staticmethod
    def define_args(parsers: ArgumentParser):
        """ define args for install command.
        """
        parser = parsers.add_parser(PluginsCMD.name)
        subparsers = parser.add_subparsers(dest='command')

        PluginsInstallCMD.define_args(subparsers)
        PluginsUninstallCMD.define_args(subparsers)
        PluginsListCMD.define_args(subparsers)

        parser.set_defaults(func=subparser_func)

    def execute(self):
        print(self.args)
        if self.args.command == PluginsInstallCMD.name:
            PluginsInstallCMD.execute(self.args)
        if self.args.command == PluginsUninstallCMD.name:
            PluginsUninstallCMD.execute(self.args)
        if self.args.command == PluginsListCMD.name:
            PluginsListCMD.execute(self.args)


class PluginsInstallCMD(PluginsCMD):
    name = 'install'

    @staticmethod
    def define_args(parsers: ArgumentParser):
        install = parsers.add_parser(PluginsInstallCMD.name)
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

    @staticmethod
    def execute(args):
        plugins_manager.install_plugins(
            list(args.package),
            index_url=args.index_url,
            force_update=args.force_update)


class PluginsUninstallCMD(PluginsCMD):
    name = 'uninstall'

    @staticmethod
    def define_args(parsers: ArgumentParser):
        install = parsers.add_parser(PluginsUninstallCMD.name)
        install.add_argument(
            'package',
            type=str,
            nargs='+',
            default=None,
            help='Name of the package to be installed.')
        install.add_argument(
            '--yes',
            '-y',
            type=str,
            default=False,
            help='Base URL of the Python Package Index.')

    @staticmethod
    def execute(args):
        plugins_manager.uninstall_plugins(list(args.package), is_yes=args.yes)


class PluginsListCMD(PluginsCMD):
    name = 'list'

    @staticmethod
    def define_args(parsers: ArgumentParser):
        install = parsers.add_parser(PluginsListCMD.name)
        install.add_argument(
            '--all',
            '-a',
            type=str,
            default=None,
            help='Show all of the plugins including those not installed.')

    @staticmethod
    def execute(args):
        plugins_manager.list_plugins(show_all=all)
