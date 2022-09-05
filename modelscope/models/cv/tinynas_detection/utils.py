# Copyright (c) Alibaba, Inc. and its affiliates.
# The AIRDet implementation is also open-sourced by the authors, and available at https://github.com/tinyvision/AIRDet.

import importlib
import os
import sys
from os.path import dirname, join


def get_config_by_file(config_file):
    try:
        sys.path.append(os.path.dirname(config_file))
        current_config = importlib.import_module(
            os.path.basename(config_file).split('.')[0])
        exp = current_config.Config()
    except Exception:
        raise ImportError(
            "{} doesn't contains class named 'Config'".format(config_file))
    return exp


def parse_config(config_file):
    """
    get config object by file.
    Args:
        config_file (str): file path of config.
    """
    assert (config_file is not None), 'plz provide config file'
    if config_file is not None:
        return get_config_by_file(config_file)
