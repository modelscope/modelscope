# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os
from argparse import ArgumentParser
from string import Template

from modelscope.cli.base import CLICommand
from modelscope.utils.logger import get_logger

logger = get_logger(log_level=logging.WARNING)

current_path = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(current_path, 'template')


def subparser_func(args):
    """ Function which will be called for a specific sub parser.
    """
    return PipelineCMD(args)


class PipelineCMD(CLICommand):
    name = 'pipeline'

    def __init__(self, args):
        self.args = args

    @staticmethod
    def define_args(parsers: ArgumentParser):
        """ define args for create pipeline template command.
        """
        parser = parsers.add_parser(PipelineCMD.name)
        parser.add_argument(
            '-act',
            '--action',
            type=str,
            required=True,
            choices=['create'],
            help='the action of command  pipeline[create]')
        parser.add_argument(
            '-tpl',
            '--tpl_file_path',
            type=str,
            default='template.tpl',
            help='the template be selected for ModelScope[template.tpl]')
        parser.add_argument(
            '-s',
            '--save_file_path',
            type=str,
            default='./',
            help='the name of custom template be saved for ModelScope')
        parser.add_argument(
            '-f',
            '--filename',
            type=str,
            default='ms_wrapper.py',
            help='the init name of custom template be saved for ModelScope')
        parser.add_argument(
            '-t',
            '--task_name',
            type=str,
            required=True,
            help='the unique task_name for ModelScope')
        parser.add_argument(
            '-m',
            '--model_name',
            type=str,
            default='MyCustomModel',
            help='the class of model name for ModelScope')
        parser.add_argument(
            '-p',
            '--preprocessor_name',
            type=str,
            default='MyCustomPreprocessor',
            help='the class of preprocessor name for ModelScope')
        parser.add_argument(
            '-pp',
            '--pipeline_name',
            type=str,
            default='MyCustomPipeline',
            help='the class of pipeline name for ModelScope')
        parser.add_argument(
            '-config',
            '--configuration_path',
            type=str,
            default='./',
            help='the path of configuration.json for ModelScope')
        parser.set_defaults(func=subparser_func)

    def create_template(self):
        if self.args.tpl_file_path not in os.listdir(template_path):
            tpl_file_path = self.args.tpl_file_path
        else:
            tpl_file_path = os.path.join(template_path,
                                         self.args.tpl_file_path)
        if not os.path.exists(tpl_file_path):
            raise ValueError('%s not exists!' % tpl_file_path)

        save_file_path = self.args.save_file_path if self.args.save_file_path != './' else os.getcwd(
        )
        os.makedirs(save_file_path, exist_ok=True)
        if not self.args.filename.endswith('.py'):
            raise ValueError('the FILENAME must end with .py ')
        save_file_name = self.args.filename
        save_pkl_path = os.path.join(save_file_path, save_file_name)

        if not self.args.configuration_path.endswith('/'):
            self.args.configuration_path = self.args.configuration_path + '/'

        lines = []
        with open(tpl_file_path) as tpl_file:
            tpl = Template(tpl_file.read())
            lines.append(tpl.substitute(**vars(self.args)))

        with open(save_pkl_path, 'w') as save_file:
            save_file.writelines(lines)

        logger.info('>>> Configuration be saved in %s/%s' %
                    (self.args.configuration_path, 'configuration.json'))
        logger.info('>>> Task_name: %s, Created in %s' %
                    (self.args.task_name, save_pkl_path))
        logger.info('Open the file < %s >, update and run it.' % save_pkl_path)

    def execute(self):
        if self.args.action == 'create':
            self.create_template()
        else:
            raise ValueError('The parameter of action must be in [create]')
