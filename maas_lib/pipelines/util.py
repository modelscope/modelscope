# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp

import json
from maas_hub.constants import MODEL_ID_SEPARATOR
from maas_hub.file_download import model_file_download

from maas_lib.utils.constant import CONFIGFILE


# temp solution before the hub-cache is in place
def get_model_cache_dir(model_id: str, branch: str = 'master'):
    model_id_expanded = model_id.replace('/',
                                         MODEL_ID_SEPARATOR) + '.' + branch
    default_cache_dir = os.path.expanduser(os.path.join('~/.cache', 'maas'))
    return os.getenv('MAAS_CACHE',
                     os.path.join(default_cache_dir, 'hub', model_id_expanded))


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
