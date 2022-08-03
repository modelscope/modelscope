import os

from modelscope.hub.constants import (DEFAULT_MODELSCOPE_DOMAIN,
                                      DEFAULT_MODELSCOPE_GROUP,
                                      MODEL_ID_SEPARATOR,
                                      MODELSCOPE_URL_SCHEME)
from modelscope.utils.file_utils import get_default_cache_dir


def model_id_to_group_owner_name(model_id):
    if MODEL_ID_SEPARATOR in model_id:
        group_or_owner = model_id.split(MODEL_ID_SEPARATOR)[0]
        name = model_id.split(MODEL_ID_SEPARATOR)[1]
    else:
        group_or_owner = DEFAULT_MODELSCOPE_GROUP
        name = model_id
    return group_or_owner, name


def get_cache_dir():
    """
    cache dir precedence:
        function parameter > enviroment > ~/.cache/modelscope/hub
    """
    default_cache_dir = get_default_cache_dir()
    return os.getenv('MODELSCOPE_CACHE', os.path.join(default_cache_dir,
                                                      'hub'))


def get_endpoint():
    modelscope_domain = os.getenv('MODELSCOPE_DOMAIN',
                                  DEFAULT_MODELSCOPE_DOMAIN)
    return MODELSCOPE_URL_SCHEME + modelscope_domain
