# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import copy
import os

import json
import yaml

from modelscope.utils.logger import get_logger

logger = get_logger()


def setup_seed(seed):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Config(object):

    def __init__(self,
                 load=True,
                 cfg_dict=None,
                 cfg_level=None,
                 model_dir=None,
                 cfg_file_name='exp06_text_depths_vs_style.yaml'):
        if model_dir is not None and os.path.isdir(model_dir):
            self.model_dir = model_dir + '/configs'
        else:
            raise Exception(f'model_dir {model_dir} is not exist!')
        self.cfg_file_name = cfg_file_name
        self._level = 'cfg' + ('.'
                               + cfg_level if cfg_level is not None else '')
        if load:
            self.args = self._parse_args()
            logger.info('Loading config from {}.'.format(self.args.cfg_file))
            self.need_initialization = True
            cfg_base = self._initialize_cfg()
            cfg_dict = self._load_yaml(self.args)
            cfg_dict = self._merge_cfg_from_base(cfg_base, cfg_dict)
            cfg_dict = self._update_from_args(cfg_dict)
            self.cfg_dict = cfg_dict
        self._update_dict(cfg_dict)

    def _parse_args(self):
        parser = argparse.ArgumentParser(
            description=
            'Argparser for configuring [code base name to think of] codebase')
        parser.add_argument(
            '--cfg',
            dest='cfg_file',
            help='Path to the configuration file',
            default=os.path.join(self.model_dir, self.cfg_file_name))
        parser.add_argument(
            '--init_method',
            help='Initialization method, includes TCP or shared file-system',
            default='tcp://localhost:9999',
            type=str,
        )
        parser.add_argument(
            '--seed',
            type=int,
            default=8888,
            help='Need to explore for different videos')
        parser.add_argument(
            '--debug',
            action='store_true',
            default=False,
            help='Into debug information')
        parser.add_argument(
            '--input_video',
            default='demo_video/video_8800.mp4',
            help='input video for full task, or motion vector of input videos',
            type=str,
        ),
        parser.add_argument(
            '--image_path', default='', help='Single Image Input', type=str)
        parser.add_argument(
            '--sketch_path', default='', help='Single Sketch Input', type=str)
        parser.add_argument(
            '--style_image', help='Single Sketch Input', type=str)
        parser.add_argument(
            '--input_text_desc',
            default=
            'A colorful and beautiful fish swimming in a small glass bowl with \
             multicolored piece of stone, Macro Video',
            type=str,
        ),
        parser.add_argument(
            'opts',
            help='other configurations',
            default=None,
            nargs=argparse.REMAINDER)
        return parser.parse_args()

    def _path_join(self, path_list):
        path = ''
        for p in path_list:
            path += p + '/'
        return path[:-1]

    def _update_from_args(self, cfg_dict):
        args = self.args
        for var in vars(args):
            cfg_dict[var] = getattr(args, var)
        return cfg_dict

    def _initialize_cfg(self):
        if self.need_initialization:
            self.need_initialization = False
            if os.path.exists(os.path.join(self.model_dir, 'base.yaml')):
                with open(os.path.join(self.model_dir, 'base.yaml'), 'r') as f:
                    cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
            else:
                with open(os.path.join(self.model_dir, 'base.yaml'), 'r') as f:
                    cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        return cfg

    def _load_yaml(self, args, file_name=''):
        assert args.cfg_file is not None
        if not file_name == '':  # reading from base file
            with open(file_name, 'r') as f:
                cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        else:
            if os.getcwd().split('/')[-1] == args.cfg_file.split('/')[0]:
                args.cfg_file = args.cfg_file.replace(
                    os.getcwd().split('/')[-1], './')
            try:
                with open(args.cfg_file, 'r') as f:
                    cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
                    file_name = args.cfg_file
            except Exception as e:
                args.cfg_file = os.path.realpath(__file__).split(
                    '/')[-3] + '/' + args.cfg_file
                with open(args.cfg_file, 'r') as f:
                    cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
                    file_name = args.cfg_file
                print(e)

        if '_BASE_RUN' not in cfg.keys() and '_BASE_MODEL' not in cfg.keys(
        ) and '_BASE' not in cfg.keys():
            return cfg

        if '_BASE' in cfg.keys():
            if cfg['_BASE'][1] == '.':
                prev_count = cfg['_BASE'].count('..')
                cfg_base_file = self._path_join(
                    file_name.split('/')[:(-1 - cfg['_BASE'].count('..'))]
                    + cfg['_BASE'].split('/')[prev_count:])
            else:
                cfg_base_file = cfg['_BASE'].replace(
                    './',
                    args.cfg_file.replace(args.cfg_file.split('/')[-1], ''))
            cfg_base = self._load_yaml(args, cfg_base_file)
            cfg = self._merge_cfg_from_base(cfg_base, cfg)
        else:
            if '_BASE_RUN' in cfg.keys():
                if cfg['_BASE_RUN'][1] == '.':
                    prev_count = cfg['_BASE_RUN'].count('..')
                    cfg_base_file = self._path_join(
                        file_name.split('/')[:(-1 - prev_count)]
                        + cfg['_BASE_RUN'].split('/')[prev_count:])
                else:
                    cfg_base_file = cfg['_BASE_RUN'].replace(
                        './',
                        args.cfg_file.replace(
                            args.cfg_file.split('/')[-1], ''))
                cfg_base = self._load_yaml(args, cfg_base_file)
                cfg = self._merge_cfg_from_base(
                    cfg_base, cfg, preserve_base=True)
            if '_BASE_MODEL' in cfg.keys():
                if cfg['_BASE_MODEL'][1] == '.':
                    prev_count = cfg['_BASE_MODEL'].count('..')
                    cfg_base_file = self._path_join(
                        file_name.split('/')[:(
                            -1 - cfg['_BASE_MODEL'].count('..'))]
                        + cfg['_BASE_MODEL'].split('/')[prev_count:])
                else:
                    cfg_base_file = cfg['_BASE_MODEL'].replace(
                        './',
                        args.cfg_file.replace(
                            args.cfg_file.split('/')[-1], ''))
                cfg_base = self._load_yaml(args, cfg_base_file)
                cfg = self._merge_cfg_from_base(cfg_base, cfg)
        cfg = self._merge_cfg_from_command(args, cfg)
        return cfg

    def _merge_cfg_from_base(self, cfg_base, cfg_new, preserve_base=False):
        for k, v in cfg_new.items():
            if k in cfg_base.keys():
                if isinstance(v, dict):
                    self._merge_cfg_from_base(cfg_base[k], v)
                else:
                    cfg_base[k] = v
            else:
                if 'BASE' not in k or preserve_base:
                    cfg_base[k] = v
        return cfg_base

    def _merge_cfg_from_command(self, args, cfg):
        assert len(
            args.opts) % 2 == 0, 'Override list {} has odd length: {}.'.format(
                args.opts, len(args.opts))
        keys = args.opts[0::2]
        vals = args.opts[1::2]

        # maximum supported depth 3
        for idx, key in enumerate(keys):
            key_split = key.split('.')
            assert len(
                key_split
            ) <= 4, 'Key depth error. \nMaximum depth: 3\n Get depth: {}'.format(
                len(key_split))
            assert key_split[0] in cfg.keys(), 'Non-existant key: {}.'.format(
                key_split[0])
            if len(key_split) == 2:
                assert key_split[1] in cfg[
                    key_split[0]].keys(), 'Non-existant key: {}.'.format(key)
            elif len(key_split) == 3:
                assert key_split[1] in cfg[
                    key_split[0]].keys(), 'Non-existant key: {}.'.format(key)
                assert key_split[2] in cfg[key_split[0]][
                    key_split[1]].keys(), 'Non-existant key: {}.'.format(key)
            elif len(key_split) == 4:
                assert key_split[1] in cfg[
                    key_split[0]].keys(), 'Non-existant key: {}.'.format(key)
                assert key_split[2] in cfg[key_split[0]][
                    key_split[1]].keys(), 'Non-existant key: {}.'.format(key)
                assert key_split[3] in cfg[key_split[0]][key_split[1]][
                    key_split[2]].keys(), 'Non-existant key: {}.'.format(key)
            if len(key_split) == 1:
                cfg[key_split[0]] = vals[idx]
            elif len(key_split) == 2:
                cfg[key_split[0]][key_split[1]] = vals[idx]
            elif len(key_split) == 3:
                cfg[key_split[0]][key_split[1]][key_split[2]] = vals[idx]
            elif len(key_split) == 4:
                cfg[key_split[0]][key_split[1]][key_split[2]][
                    key_split[3]] = vals[idx]
        return cfg

    def _update_dict(self, cfg_dict):

        def recur(key, elem):
            if type(elem) is dict:
                return key, Config(
                    load=False,
                    cfg_dict=elem,
                    cfg_level=key,
                    model_dir=self.model_dir,
                    cfg_file_name=self.cfg_file_name)
            else:
                if type(elem) is str and elem[1:3] == 'e-':
                    elem = float(elem)
                return key, elem

        dic = dict(recur(k, v) for k, v in cfg_dict.items())
        self.__dict__.update(dic)

    def get_args(self):
        return self.args

    def __repr__(self):
        return '{}\n'.format(self.dump())

    def dump(self):
        return json.dumps(self.cfg_dict, indent=2)

    def deep_copy(self):
        return copy.deepcopy(self)
