MODELSCOPE_URL_SCHEME = 'http://'
DEFAULT_MODELSCOPE_DOMAIN = '47.94.223.21:31090'
DEFAULT_MODELSCOPE_GITLAB_DOMAIN = '101.201.119.157:31102'

DEFAULT_MODELSCOPE_GROUP = 'damo'
MODEL_ID_SEPARATOR = '/'

LOGGER_NAME = 'ModelScopeHub'


class Licenses(object):
    APACHE_V2 = 'Apache License 2.0'
    GPL = 'GPL'
    LGPL = 'LGPL'
    MIT = 'MIT'


class ModelVisibility(object):
    PRIVATE = 1
    INTERNAL = 3
    PUBLIC = 5
