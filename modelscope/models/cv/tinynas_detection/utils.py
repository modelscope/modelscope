# Copyright (c) Alibaba, Inc. and its affiliates.
# The DAMO-YOLO implementation is also open-sourced by the authors, and available
# at https://github.com/tinyvision/damo-yolo.

import importlib
import os
import shutil
import sys
import tempfile
from os.path import dirname, join

from easydict import EasyDict


def parse_config(filename):
    filename = str(filename)
    if filename.endswith('.py'):
        with tempfile.TemporaryDirectory() as temp_config_dir:
            shutil.copyfile(filename, join(temp_config_dir, '_tempconfig.py'))
            sys.path.insert(0, temp_config_dir)
            mod = importlib.import_module('_tempconfig')
            sys.path.pop(0)
            cfg_dict = EasyDict({
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith('__')
            })
            # delete imported module
            del sys.modules['_tempconfig']
    else:
        raise IOError('Only .py type are supported now!')

    return cfg_dict
