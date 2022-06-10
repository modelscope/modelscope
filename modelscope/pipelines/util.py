# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
from typing import List, Union

import json
from maas_hub.file_download import model_file_download

from modelscope.utils.constant import CONFIGFILE


def is_model_name(model: Union[str, List]):
    """ whether model is a valid modelhub path
    """

    def is_model_name_impl(model):
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

    if isinstance(model, str):
        return is_model_name_impl(model)
    else:
        results = [is_model_name_impl(m) for m in model]
        all_true = all(results)
        any_true = any(results)
        if any_true and not all_true:
            raise ValueError('some model are hub address, some are not')

        return all_true
