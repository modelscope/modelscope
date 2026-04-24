# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from pathlib import Path

MODELSCOPE_URL_SCHEME = 'https://'
DEFAULT_MODELSCOPE_DOMAIN = 'www.modelscope.cn'
DEFAULT_MODELSCOPE_INTL_DOMAIN = 'www.modelscope.ai'
DEFAULT_MODELSCOPE_DATA_ENDPOINT = MODELSCOPE_URL_SCHEME + DEFAULT_MODELSCOPE_DOMAIN
DEFAULT_MODELSCOPE_INTL_DATA_ENDPOINT = MODELSCOPE_URL_SCHEME + DEFAULT_MODELSCOPE_INTL_DOMAIN
MODELSCOPE_PARALLEL_DOWNLOAD_THRESHOLD_MB = int(
    os.environ.get('MODELSCOPE_PARALLEL_DOWNLOAD_THRESHOLD_MB', 500))
MODELSCOPE_DOWNLOAD_PARALLELS = int(
    os.environ.get('MODELSCOPE_DOWNLOAD_PARALLELS', 1))
DEFAULT_MODELSCOPE_GROUP = 'damo'
MODEL_ID_SEPARATOR = '/'
FILE_HASH = 'Sha256'
LOGGER_NAME = 'ModelScopeHub'
DEFAULT_CREDENTIALS_PATH = Path.home().joinpath('.modelscope', 'credentials')
MODELSCOPE_CREDENTIALS_PATH = os.environ.get(
    'MODELSCOPE_CREDENTIALS_PATH', DEFAULT_CREDENTIALS_PATH.as_posix())
REQUESTS_API_HTTP_METHOD = ['get', 'head', 'post', 'put', 'patch', 'delete']
API_HTTP_CLIENT_TIMEOUT = 60
API_HTTP_CLIENT_MAX_RETRIES = int(
    os.environ.get('API_HTTP_CLIENT_MAX_RETRIES', 5))

# Application-level retry for blob upload
UPLOAD_BLOB_MAX_RETRIES = int(os.environ.get('UPLOAD_BLOB_MAX_RETRIES', 5))
UPLOAD_BLOB_RETRY_BACKOFF = int(os.environ.get('UPLOAD_BLOB_RETRY_BACKOFF', 2))
UPLOAD_BLOB_RETRY_MAX_WAIT = int(
    os.environ.get('UPLOAD_BLOB_RETRY_MAX_WAIT', 60))

# Failed file retry within upload_folder
UPLOAD_FAILED_FILE_MAX_RETRIES = int(
    os.environ.get('UPLOAD_FAILED_FILE_MAX_RETRIES', 3))

# Per-socket-operation timeout for blob upload (not total transfer time).
# Each individual socket send/recv must complete within this limit.
# Safe for any file size: a 100GB upload's total time can exceed this,
# but each chunk I/O operation finishes in milliseconds.
UPLOAD_BLOB_TIMEOUT = (30,
                       int(
                           os.environ.get('UPLOAD_BLOB_TIMEOUT_SECONDS',
                                          3600)))

# Methods that are safe for urllib3 transport-level retry.
# PUT is excluded because blob uploads use streaming data that cannot be replayed.
UPLOAD_RETRY_ALLOWED_METHODS = frozenset(
    os.environ.get('UPLOAD_RETRY_ALLOWED_METHODS',
                   'GET,HEAD,DELETE,OPTIONS,TRACE').split(','))

API_RESPONSE_FIELD_DATA = 'Data'
API_FILE_DOWNLOAD_RETRY_TIMES = 5
API_FILE_DOWNLOAD_TIMEOUT = 60
API_FILE_DOWNLOAD_CHUNK_SIZE = 1024 * 1024 * 1
API_RESPONSE_FIELD_GIT_ACCESS_TOKEN = 'AccessToken'
API_RESPONSE_FIELD_USERNAME = 'Username'
API_RESPONSE_FIELD_EMAIL = 'Email'
API_RESPONSE_FIELD_MESSAGE = 'Message'
MODELSCOPE_CLOUD_ENVIRONMENT = 'MODELSCOPE_ENVIRONMENT'
MODELSCOPE_CLOUD_USERNAME = 'MODELSCOPE_USERNAME'
MODELSCOPE_SDK_DEBUG = 'MODELSCOPE_SDK_DEBUG'
MODELSCOPE_PREFER_AI_SITE = 'MODELSCOPE_PREFER_AI_SITE'
MODELSCOPE_DOMAIN = 'MODELSCOPE_DOMAIN'
MODELSCOPE_ENABLE_DEFAULT_HASH_VALIDATION = 'MODELSCOPE_ENABLE_DEFAULT_HASH_VALIDATION'
ONE_YEAR_SECONDS = 24 * 365 * 60 * 60
MODELSCOPE_REQUEST_ID = 'X-Request-ID'
TEMPORARY_FOLDER_NAME = '._____temp'
DEFAULT_MAX_WORKERS = int(
    os.getenv('DEFAULT_MAX_WORKERS', min(8,
                                         os.cpu_count() + 4)))
DEFAULT_SKILLS_DIR = os.path.join(os.path.expanduser('~'), '.agents', 'skills')

# Upload check env
UPLOAD_MAX_FILE_SIZE = int(
    os.environ.get('UPLOAD_MAX_FILE_SIZE_MB', 100 * 1024)) * 1024 * 1024
UPLOAD_SIZE_THRESHOLD_TO_ENFORCE_LFS = int(
    os.environ.get('UPLOAD_SIZE_THRESHOLD_TO_ENFORCE_LFS', 1 * 1024 * 1024))
UPLOAD_MAX_FILE_COUNT = int(os.environ.get('UPLOAD_MAX_FILE_COUNT', 100_000))
UPLOAD_MAX_FILE_COUNT_IN_DIR = int(
    os.environ.get('UPLOAD_MAX_FILE_COUNT_IN_DIR', 50_000))
UPLOAD_NORMAL_FILE_SIZE_TOTAL_LIMIT = int(
    os.environ.get('UPLOAD_NORMAL_FILE_SIZE_TOTAL_LIMIT', 500 * 1024 * 1024))
UPLOAD_COMMIT_BATCH_SIZE = int(os.environ.get('UPLOAD_COMMIT_BATCH_SIZE', 256))
UPLOAD_VALIDATE_BLOB_BATCH_SIZE = int(
    os.environ.get('UPLOAD_VALIDATE_BLOB_BATCH_SIZE', 64))
UPLOAD_ADAPTIVE_BATCH_SIZE = os.environ.get('UPLOAD_ADAPTIVE_BATCH_SIZE',
                                            'true').lower() == 'true'

# ReAct progressive retry fallback
UPLOAD_REACT_ENABLED = os.environ.get('UPLOAD_REACT_ENABLED',
                                      'true').lower() == 'true'
UPLOAD_REACT_ROUND2_BASE_DELAY = int(
    os.environ.get('UPLOAD_REACT_ROUND2_BASE_DELAY', 2))
UPLOAD_REACT_ROUND3_FILE_DELAY = int(
    os.environ.get('UPLOAD_REACT_ROUND3_FILE_DELAY', 30))
UPLOAD_REACT_BACKOFF_MAX_EXPONENT = int(
    os.environ.get('UPLOAD_REACT_BACKOFF_MAX_EXPONENT', 5))
UPLOAD_REACT_MAX_DELAY = int(os.environ.get('UPLOAD_REACT_MAX_DELAY', 120))

UPLOAD_BLOB_TQDM_DISABLE_THRESHOLD = 5 * 1024 * 1024
UPLOAD_USE_CACHE = os.environ.get('UPLOAD_USE_CACHE', 'true').lower() == 'true'


MODELSCOPE_ASCII = r"""
 _   .-')                _ .-') _     ('-.             .-')                              _ (`-.    ('-.
( '.( OO )_             ( (  OO) )  _(  OO)           ( OO ).                           ( (OO  ) _(  OO)
 ,--.   ,--.).-'),-----. \     .'_ (,------.,--.     (_)---\_)   .-----.  .-'),-----.  _.`     \(,------.
 |   `.'   |( OO'  .-.  ',`'--..._) |  .---'|  |.-') /    _ |   '  .--./ ( OO'  .-.  '(__...--'' |  .---'
 |         |/   |  | |  ||  |  \  ' |  |    |  | OO )\  :` `.   |  |('-. /   |  | |  | |  /  | | |  |
 |  |'.'|  |\_) |  |\|  ||  |   ' |(|  '--. |  |`-' | '..`''.) /_) |OO  )\_) |  |\|  | |  |_.' |(|  '--.
 |  |   |  |  \ |  | |  ||  |   / : |  .--'(|  '---.'.-._)   \ ||  |`-'|   \ |  | |  | |  .___.' |  .--'
 |  |   |  |   `'  '-'  '|  '--'  / |  `---.|      | \       /(_'  '--'\    `'  '-'  ' |  |      |  `---.
 `--'   `--'     `-----' `-------'  `------'`------'  `-----'    `-----'      `-----'  `--'      `------'
"""# noqa


class Licenses(object):
    APACHE_V2 = 'Apache License 2.0'
    GPL_V2 = 'GPL-2.0'
    GPL_V3 = 'GPL-3.0'
    LGPL_V2_1 = 'LGPL-2.1'
    LGPL_V3 = 'LGPL-3.0'
    AFL_V3 = 'AFL-3.0'
    ECL_V2 = 'ECL-2.0'
    MIT = 'MIT'

    @classmethod
    def to_list(cls):
        return [
            cls.APACHE_V2,
            cls.GPL_V2,
            cls.GPL_V3,
            cls.LGPL_V2_1,
            cls.LGPL_V3,
            cls.AFL_V3,
            cls.ECL_V2,
            cls.MIT,
        ]


class ModelVisibility(object):
    PRIVATE = 1
    INTERNAL = 3
    PUBLIC = 5


class DatasetVisibility(object):
    PRIVATE = 1
    INTERNAL = 3
    PUBLIC = 5


class Visibility(object):
    PRIVATE = 'private'
    INTERNAL = 'internal'
    PUBLIC = 'public'


VisibilityMap = {
    ModelVisibility.PRIVATE: Visibility.PRIVATE,
    ModelVisibility.INTERNAL: Visibility.INTERNAL,
    ModelVisibility.PUBLIC: Visibility.PUBLIC
}


class SortKey(object):
    DEFAULT = 'default'
    DOWNLOADS = 'downloads'
    LIKES = 'likes'
    LAST_MODIFIED = 'last_modified'


VALID_SORT_KEYS = {
    SortKey.DEFAULT,
    SortKey.DOWNLOADS,
    SortKey.LIKES,
    SortKey.LAST_MODIFIED,
}
