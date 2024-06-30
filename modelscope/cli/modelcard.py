# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os
import shutil
import tempfile
from argparse import ArgumentParser
from string import Template

from modelscope.cli.base import CLICommand
from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.hub.utils.utils import get_endpoint
from modelscope.utils.logger import get_logger

logger = get_logger(log_level=logging.WARNING)

current_path = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(current_path, 'template')


def subparser_func(args):
    """ Function which will be called for a specific sub parser.
    """
    return ModelCardCMD(args)


class ModelCardCMD(CLICommand):
    name = 'modelcard'

    def __init__(self, args):
        self.args = args
        self.api = HubApi()
        if args.access_token:
            self.api.login(args.access_token)
        self.model_id = os.path.join(
            self.args.group_id, self.args.model_id
        ) if '/' not in self.args.model_id else self.args.model_id
        self.url = os.path.join(get_endpoint(), self.model_id)

    @staticmethod
    def define_args(parsers: ArgumentParser):
        """ define args for create or upload modelcard command.
        """
        parser = parsers.add_parser(ModelCardCMD.name, aliases=['model'])
        parser.add_argument(
            '-tk',
            '--access_token',
            type=str,
            required=False,
            help='the certification of visit ModelScope')
        parser.add_argument(
            '-act',
            '--action',
            type=str,
            required=True,
            choices=['create', 'upload', 'download'],
            help='the action of api ModelScope[create, upload]')
        parser.add_argument(
            '-gid',
            '--group_id',
            type=str,
            default='damo',
            help='the group name of ModelScope, eg, damo')
        parser.add_argument(
            '-mid',
            '--model_id',
            type=str,
            required=True,
            help='the model name of ModelScope')
        parser.add_argument(
            '-vis',
            '--visibility',
            type=int,
            default=5,
            help=
            'the visibility of ModelScope[PRIVATE: 1, INTERNAL:3, PUBLIC:5]')
        parser.add_argument(
            '-lic',
            '--license',
            type=str,
            default='Apache License 2.0',
            help='the license of visit ModelScope[Apache License 2.0|'
            'GPL-2.0|GPL-3.0|LGPL-2.1|LGPL-3.0|AFL-3.0|ECL-2.0|MIT]')
        parser.add_argument(
            '-ch',
            '--chinese_name',
            type=str,
            default='这是我的第一个模型',
            help='the chinese name of ModelScope')
        parser.add_argument(
            '-md',
            '--model_dir',
            type=str,
            default='.',
            help='the model_dir of configuration.json')
        parser.add_argument(
            '-vt',
            '--version_tag',
            type=str,
            default=None,
            help='the tag of uploaded model')
        parser.add_argument(
            '-vi',
            '--version_info',
            type=str,
            default=None,
            help='the info of uploaded model')
        parser.set_defaults(func=subparser_func)

    def create_model(self):
        from modelscope.hub.constants import Licenses, ModelVisibility
        visibilities = [
            getattr(ModelVisibility, attr) for attr in dir(ModelVisibility)
            if not attr.startswith('__')
        ]
        if self.args.visibility not in visibilities:
            raise ValueError('The access_token must in %s!' % visibilities)
        licenses = [
            getattr(Licenses, attr) for attr in dir(Licenses)
            if not attr.startswith('__')
        ]
        if self.args.license not in licenses:
            raise ValueError('The license must in %s!' % licenses)
        try:
            self.api.get_model(self.model_id)
        except Exception as e:
            logger.info('>>> %s' % type(e))
            self.api.create_model(
                model_id=self.model_id,
                visibility=self.args.visibility,
                license=self.args.license,
                chinese_name=self.args.chinese_name,
            )
        self.pprint()

    def get_model_url(self):
        return self.api.get_model_url(self.model_id)

    def push_model(self, tpl_dir='readme.tpl'):
        from modelscope.hub.repository import Repository
        if self.args.version_tag and self.args.version_info:
            clone_dir = tempfile.TemporaryDirectory().name
            repo = Repository(clone_dir, clone_from=self.model_id)
            repo.tag_and_push(self.args.version_tag, self.args.version_info)
            shutil.rmtree(clone_dir)
        else:
            cfg_file = os.path.join(self.args.model_dir, 'README.md')
            if not os.path.exists(cfg_file):
                with open(os.path.join(template_path,
                                       tpl_dir)) as tpl_file_path:
                    tpl = Template(tpl_file_path.read())
                    f = open(cfg_file, 'w')
                    f.write(tpl.substitute(model_id=self.model_id))
                    f.close()
            self.api.push_model(
                model_id=self.model_id,
                model_dir=self.args.model_dir,
                visibility=self.args.visibility,
                license=self.args.license,
                chinese_name=self.args.chinese_name)
        self.pprint()

    def pprint(self):
        logger.info('>>> Clone the model_git < %s >, commit and push it.'
                    % self.get_model_url())
        logger.info('>>> Open the url < %s >, check and read it.' % self.url)
        logger.info('>>> Visit the model_id < %s >, download and run it.'
                    % self.model_id)

    def execute(self):
        if self.args.action == 'create':
            self.create_model()
        elif self.args.action == 'upload':
            self.push_model()
        elif self.args.action == 'download':
            snapshot_download(
                self.model_id,
                cache_dir=self.args.model_dir,
                revision=self.args.version_tag)
        else:
            raise ValueError(
                'The parameter of action must be in [create, upload]')
