# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp

import json
from maas_hub.file_download import model_file_download

from modelscope.utils.constant import CONFIGFILE


def is_model_name(model):
    if osp.exists(model):
        if osp.exists(osp.join(model, CONFIGFILE)):
            return True
        else:
            return False
    else:
        # try:
        #     cfg_file = model_file_download(model, CONFIGFILE)
        # except Exception:
        #     cfg_file = None
        # TODO @wenmeng.zwm use exception instead of
        # following tricky logic
        cfg_file = model_file_download(model, CONFIGFILE)
        with open(cfg_file, 'r') as infile:
            cfg = json.load(infile)
        if 'Code' in cfg:
            return False
        else:
            return True
