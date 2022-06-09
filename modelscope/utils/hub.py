# Copyright (c) Alibaba, Inc. and its affiliates.

import os

from maas_hub.constants import MODEL_ID_SEPARATOR


# temp solution before the hub-cache is in place
def get_model_cache_dir(model_id: str, branch: str = 'master'):
    model_id_expanded = model_id.replace('/',
                                         MODEL_ID_SEPARATOR) + '.' + branch
    default_cache_dir = os.path.expanduser(os.path.join('~/.cache', 'maas'))
    return os.getenv('MAAS_CACHE',
                     os.path.join(default_cache_dir, 'hub', model_id_expanded))
