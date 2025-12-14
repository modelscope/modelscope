# Copyright (c) Alibaba, Inc. and its affiliates.
# yapf: disable

import datetime
import fnmatch
import functools
import io
import os
import pickle
import platform
import re
import shutil
import tempfile
import time
import uuid
import warnings
from collections import defaultdict
from http import HTTPStatus
from http.cookiejar import CookieJar
from os.path import expanduser
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import urlencode

import json
import requests
from requests import Session
from requests.adapters import HTTPAdapter, Retry
from requests.exceptions import HTTPError
from tqdm.auto import tqdm

from modelscope.hub.constants import (API_HTTP_CLIENT_MAX_RETRIES,
                                      API_HTTP_CLIENT_TIMEOUT,
                                      API_RESPONSE_FIELD_DATA,
                                      API_RESPONSE_FIELD_EMAIL,
                                      API_RESPONSE_FIELD_GIT_ACCESS_TOKEN,
                                      API_RESPONSE_FIELD_MESSAGE,
                                      API_RESPONSE_FIELD_USERNAME,
                                      DEFAULT_MAX_WORKERS,
                                      DEFAULT_MODELSCOPE_INTL_DOMAIN,
                                      MODELSCOPE_CLOUD_ENVIRONMENT,
                                      MODELSCOPE_CLOUD_USERNAME,
                                      MODELSCOPE_CREDENTIALS_PATH,
                                      MODELSCOPE_DOMAIN,
                                      MODELSCOPE_PREFER_AI_SITE,
                                      MODELSCOPE_REQUEST_ID,
                                      MODELSCOPE_URL_SCHEME, ONE_YEAR_SECONDS,
                                      REQUESTS_API_HTTP_METHOD,
                                      TEMPORARY_FOLDER_NAME,
                                      UPLOAD_BLOB_TQDM_DISABLE_THRESHOLD,
                                      UPLOAD_COMMIT_BATCH_SIZE,
                                      UPLOAD_MAX_FILE_COUNT,
                                      UPLOAD_MAX_FILE_COUNT_IN_DIR,
                                      UPLOAD_MAX_FILE_SIZE,
                                      UPLOAD_NORMAL_FILE_SIZE_TOTAL_LIMIT,
                                      UPLOAD_SIZE_THRESHOLD_TO_ENFORCE_LFS,
                                      VALID_SORT_KEYS, DatasetVisibility,
                                      Licenses, ModelVisibility, Visibility,
                                      VisibilityMap)
from modelscope.hub.errors import (InvalidParameter, NotExistError,
                                   NotLoginException, RequestError,
                                   datahub_raise_on_error,
                                   handle_http_post_error,
                                   handle_http_response, is_ok,
                                   raise_for_http_status, raise_on_error)
from modelscope.hub.git import GitCommandWrapper
from modelscope.hub.info import DatasetInfo, ModelInfo
from modelscope.hub.repository import Repository
from modelscope.hub.utils.aigc import AigcModel
from modelscope.hub.utils.utils import (add_content_to_file, get_domain,
                                        get_endpoint, get_readable_folder_size,
                                        get_release_datetime, is_env_true,
                                        model_id_to_group_owner_name)
from modelscope.utils.constant import (DEFAULT_DATASET_REVISION,
                                       DEFAULT_MODEL_REVISION,
                                       DEFAULT_REPOSITORY_REVISION,
                                       MASTER_MODEL_BRANCH, META_FILES_FORMAT,
                                       REPO_TYPE_DATASET, REPO_TYPE_MODEL,
                                       REPO_TYPE_SUPPORT, ConfigFields,
                                       DatasetFormations, DatasetMetaFormats,
                                       DownloadChannel, DownloadMode,
                                       Frameworks, ModelFile, Tasks,
                                       VirgoDatasetConfig)
from modelscope.utils.file_utils import get_file_hash, get_file_size
from modelscope.utils.logger import get_logger
from modelscope.utils.repo_utils import (DATASET_LFS_SUFFIX,
                                         DEFAULT_IGNORE_PATTERNS,
                                         MODEL_LFS_SUFFIX,
                                         CommitHistoryResponse, CommitInfo,
                                         CommitOperation, CommitOperationAdd,
                                         RepoUtils)
from modelscope.utils.thread_utils import thread_executor

logger = get_logger()


class HubApi:
    """Model hub api interface.
    """

    def __init__(self,
                 endpoint: Optional[str] = None,
                 timeout=API_HTTP_CLIENT_TIMEOUT,
                 max_retries=API_HTTP_CLIENT_MAX_RETRIES):
        """The ModelScope HubApi。

        Args:
            endpoint (str, optional): The modelscope server http|https address. Defaults to None.
        """
        self.endpoint = endpoint if endpoint is not None else get_endpoint()
        self.headers = {'user-agent': ModelScopeConfig.get_user_agent()}
        self.session = Session()
        retry = Retry(
            total=max_retries,
            read=2,
            connect=2,
            backoff_factor=1,
            status_forcelist=(500, 502, 503, 504),
            respect_retry_after_header=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        # set http timeout
        for method in REQUESTS_API_HTTP_METHOD:
            setattr(
                self.session, method,
                functools.partial(
                    getattr(self.session, method),
                    timeout=timeout))

        self.upload_checker = UploadingCheck()

    def _get_cookies(self, access_token: str):
        """
        Get jar cookies for authentication from access_token.

        Args:
            access_token (str): user access token on ModelScope.

        Returns:
            jar (CookieJar): cookies for authentication.
        """
        from requests.cookies import RequestsCookieJar
        from urllib.parse import urlparse

        domain: str = urlparse(self.endpoint).netloc if self.endpoint else get_domain()

        jar = RequestsCookieJar()
        jar.set('m_session_id',
                access_token,
                domain=domain,
                path='/')
        return jar

    def get_cookies(self, access_token, cookies_required: Optional[bool] = False):
        """
        Get cookies for authentication from local cache or access_token.

        Args:
            access_token (str): user access token on ModelScope
            cookies_required (bool): whether to raise error if no cookies found, defaults to `False`.

        Returns:
            cookies (CookieJar): cookies for authentication.

        Raises:
            ValueError: If no credentials found and cookies_required is True.
        """
        if access_token:
            cookies = self._get_cookies(access_token=access_token)
        else:
            cookies = ModelScopeConfig.get_cookies()

        if cookies is None and cookies_required:
            raise ValueError(
                'No credentials found.'
                'You can pass the `--token` argument, '
                'or use HubApi().login(access_token=`your_sdk_token`). '
                'Your token is available at https://modelscope.cn/my/myaccesstoken'
            )

        return cookies

    def login(
            self,
            access_token: Optional[str] = None,
            endpoint: Optional[str] = None
    ):
        """Login with your SDK access token, which can be obtained from
           https://www.modelscope.cn user center.

        Args:
            access_token (str): user access token on modelscope, set this argument or set `MODELSCOPE_API_TOKEN`.
            If neither of the tokens exist, login will directly return.
            endpoint: the endpoint to use, default to None to use endpoint specified in the class

        Returns:
            cookies: to authenticate yourself to ModelScope open-api
            git_token: token to access your git repository.

        Note:
            You only have to login once within 30 days.
        """
        if access_token is None:
            access_token = os.environ.get('MODELSCOPE_API_TOKEN')
        if not access_token:
            return None, None
        if not endpoint:
            endpoint = self.endpoint
        path = f'{endpoint}/api/v1/login'
        r = self.session.post(
            path,
            json={'AccessToken': access_token},
            headers=self.builder_headers(self.headers))
        raise_for_http_status(r)
        d = r.json()
        raise_on_error(d)

        token = d[API_RESPONSE_FIELD_DATA][API_RESPONSE_FIELD_GIT_ACCESS_TOKEN]
        cookies = r.cookies

        # save token and cookie
        ModelScopeConfig.save_token(token)
        ModelScopeConfig.save_cookies(cookies)
        ModelScopeConfig.save_user_info(
            d[API_RESPONSE_FIELD_DATA][API_RESPONSE_FIELD_USERNAME],
            d[API_RESPONSE_FIELD_DATA][API_RESPONSE_FIELD_EMAIL])

        return d[API_RESPONSE_FIELD_DATA][
            API_RESPONSE_FIELD_GIT_ACCESS_TOKEN], cookies

    def create_model(self,
                     model_id: str,
                     visibility: Optional[int] = ModelVisibility.PUBLIC,
                     license: Optional[str] = Licenses.APACHE_V2,
                     chinese_name: Optional[str] = None,
                     original_model_id: Optional[str] = '',
                     endpoint: Optional[str] = None,
                     token: Optional[str] = None,
                     aigc_model: Optional['AigcModel'] = None) -> str:
        """Create model repo at ModelScope Hub.

        Args:
            model_id (str): The model id in format {owner}/{name}
            visibility (int, optional): visibility of the model(1-private, 5-public), default 5.
            license (str, optional): license of the model, default apache-2.0.
            chinese_name (str, optional): chinese name of the model.
            original_model_id (str, optional): the base model id which this model is trained from
            endpoint: the endpoint to use, default to None to use endpoint specified in the class
            token (str, optional): access token for authentication
            aigc_model (AigcModel, optional): AigcModel instance for AIGC model creation.
                If provided, will create an AIGC model with automatic file upload.
                Refer to modelscope.hub.utils.aigc.AigcModel for details.

        Returns:
            str: URL of the created model repository

        Raises:
            InvalidParameter: If model_id is invalid or required AIGC parameters are missing.
            ValueError: If not login.

        Note:
            model_id = {owner}/{name}
        """
        if model_id is None:
            raise InvalidParameter('model_id is required!')
        # Get cookies for authentication.
        cookies = self.get_cookies(access_token=token, cookies_required=True)
        if not endpoint:
            endpoint = self.endpoint

        owner_or_group, name = model_id_to_group_owner_name(model_id)

        # Base body configuration
        body = {
            'Path': owner_or_group,
            'Name': name,
            'ChineseName': chinese_name,
            'Visibility': visibility,
            'License': license,
            'OriginalModelId': original_model_id,
            'TrainId': os.environ.get('MODELSCOPE_TRAIN_ID', '')
        }

        # Set path based on model type
        if aigc_model is not None:
            # Use AIGC model endpoint
            path = f'{endpoint}/api/v1/models/aigc'
            # Best-effort pre-upload weights so server recognizes sha256 (use existing cookies)
            aigc_model.preupload_weights(cookies=cookies, headers=self.builder_headers(self.headers), endpoint=endpoint)

            # Add AIGC-specific fields to body
            body.update({
                'TagShowName': aigc_model.tag,
                'CoverImages': aigc_model.cover_images,
                'AigcType': aigc_model.aigc_type,
                'TagDescription': aigc_model.description,
                'VisionFoundation': aigc_model.base_model_type,
                'BaseModel': aigc_model.base_model_id or original_model_id,
                'WeightsName': aigc_model.weight_filename,
                'WeightsSha256': aigc_model.weight_sha256,
                'WeightsSize': aigc_model.weight_size,
                'ModelPath': aigc_model.model_path,
                'TriggerWords': aigc_model.trigger_words,
                'ModelSource': aigc_model.model_source,
            })

            if aigc_model.official_tags:
                body['OfficialTags'] = aigc_model.official_tags

        else:
            # Use regular model endpoint
            path = f'{endpoint}/api/v1/models'

        headers = self.builder_headers(self.headers)

        intl_end = DEFAULT_MODELSCOPE_INTL_DOMAIN.split('.')[-1]
        if endpoint.rstrip('/').endswith(f'.{intl_end}'):
            headers['X-Modelscope-Accept-Language'] = 'en_US'
        r = self.session.post(
            path,
            json=body,
            cookies=cookies,
            headers=headers)
        raise_for_http_status(r)
        d = r.json()
        raise_on_error(d)
        model_repo_url = f'{endpoint}/models/{model_id}'

        # Upload model files for AIGC models
        if aigc_model is not None:
            aigc_model.upload_to_repo(self, model_id, token)

        return model_repo_url

    def create_model_tag(self,
                         model_id: str,
                         tag_name: str,
                         endpoint: Optional[str] = None,
                         token: Optional[str] = None,
                         aigc_model: Optional['AigcModel'] = None) -> str:
        """Create a tag for a model at ModelScope Hub.

        Args:
            model_id (str): The model id in format {owner}/{name}
            tag_name (str): The tag name (e.g., "v1.0.0")
            endpoint: the endpoint to use, default to None to use endpoint specified in the class
            token (str, optional): access token for authentication
            aigc_model (AigcModel, optional): AigcModel instance for AIGC model tag creation.
                If provided, will create an AIGC model tag with automatic parameters.
                Refer to modelscope.hub.utils.aigc.AigcModel for details.

        Returns:
            str: URL of the created tag

        Raises:
            InvalidParameter: If model_id, tag_name, ref, or description is invalid.
            ValueError: If not login.

        Note:
            model_id = {owner}/{name}
        """
        if model_id is None:
            raise InvalidParameter('model_id is required!')
        if tag_name is None:
            raise InvalidParameter('tag_name is required!')
        if tag_name.lower() in ['main', 'master']:
            raise InvalidParameter(
                f'tag_name "{tag_name}" is not allowed. '
                f'Please use a different tag name (e.g., "v1.0", "v1.1", "latest"). '
                f'Reserved names: main, master'
            )

        # Get cookies for authentication.
        cookies = self.get_cookies(access_token=token, cookies_required=True)
        if not endpoint:
            endpoint = self.endpoint

        owner_or_group, name = model_id_to_group_owner_name(model_id)

        # Set path and body based on model type
        if aigc_model is not None:
            # Use AIGC model tag endpoint
            path = f'{endpoint}/api/v1/models/aigc/repo/tag'
            aigc_model.preupload_weights(cookies=cookies, headers=self.builder_headers(self.headers), endpoint=endpoint)

            # Base body for AIGC model tag
            body = {
                'CoverImages': aigc_model.cover_images,
                'Name': name,
                'Path': owner_or_group,
                'TagShowName': tag_name,
                'WeightsName': aigc_model.weight_filename,
                'WeightsSha256': aigc_model.weight_sha256,
                'WeightsSize': aigc_model.weight_size,
                'TriggerWords': aigc_model.trigger_words
            }

        else:
            # Use regular model tag endpoint
            path = f'{endpoint}/api/v1/models/{model_id}/repo/tag'
            revision = 'master'
            body = {
                'TagName': tag_name,
                'Ref': revision
            }

        r = self.session.post(
            path,
            json=body,
            cookies=cookies,
            headers=self.builder_headers(self.headers))

        raise_for_http_status(r)
        d = r.json()
        raise_on_error(d)

        tag_url = f'{endpoint}/models/{model_id}/tags/{tag_name}'
        return tag_url

    def delete_model(self, model_id: str, endpoint: Optional[str] = None):
        """Delete model_id from ModelScope.

        Args:
            model_id (str): The model id.
            endpoint: the endpoint to use, default to None to use endpoint specified in the class

        Raises:
            ValueError: If not login.

        Note:
            model_id = {owner}/{name}
        """
        cookies = ModelScopeConfig.get_cookies()
        if not endpoint:
            endpoint = self.endpoint
        if cookies is None:
            raise ValueError('Token does not exist, please login first.')
        path = f'{endpoint}/api/v1/models/{model_id}'

        r = self.session.delete(path,
                                cookies=cookies,
                                headers=self.builder_headers(self.headers))
        raise_for_http_status(r)
        raise_on_error(r.json())

    def get_model_url(self, model_id: str, endpoint: Optional[str] = None):
        if not endpoint:
            endpoint = self.endpoint
        return f'{endpoint}/api/v1/models/{model_id}.git'

    def get_model(
            self,
            model_id: str,
            revision: Optional[str] = DEFAULT_MODEL_REVISION,
            endpoint: Optional[str] = None
    ) -> str:
        """Get model information at ModelScope

        Args:
            model_id (str): The model id.
            revision (str optional): revision of model.
            endpoint: the endpoint to use, default to None to use endpoint specified in the class

        Returns:
            The model detail information.

        Raises:
            NotExistError: If the model is not exist, will throw NotExistError

        Note:
            model_id = {owner}/{name}
        """
        cookies = ModelScopeConfig.get_cookies()
        owner_or_group, name = model_id_to_group_owner_name(model_id)
        if not endpoint:
            endpoint = self.endpoint

        if revision:
            path = f'{endpoint}/api/v1/models/{owner_or_group}/{name}?Revision={revision}'
        else:
            path = f'{endpoint}/api/v1/models/{owner_or_group}/{name}'

        r = self.session.get(path, cookies=cookies,
                             headers=self.builder_headers(self.headers))
        handle_http_response(r, logger, cookies, model_id)
        if r.status_code == HTTPStatus.OK:
            if is_ok(r.json()):
                return r.json()[API_RESPONSE_FIELD_DATA]
            else:
                raise NotExistError(r.json()[API_RESPONSE_FIELD_MESSAGE])
        else:
            raise_for_http_status(r)

    def get_endpoint_for_read(self,
                              repo_id: str,
                              *,
                              repo_type: Optional[str] = None) -> str:
        """Get proper endpoint for read operation (such as download, list etc.)
        1. If user has set MODELSCOPE_DOMAIN, construct endpoint with user-specified domain.
           If the repo does not exist on that endpoint, throw 404 error, otherwise return the endpoint.
        2. If domain is not set, check existence of repo in cn-site and ai-site (intl version) respectively.
           Checking order is determined by MODELSCOPE_PREFER_AI_SITE.
             a. if MODELSCOPE_PREFER_AI_SITE is not set ,check cn-site first before ai-site (intl version)
             b. otherwise check ai-site before cn-site
           return the endpoint with which the given repo_id exists.
           if neither exists, throw 404 error
        """
        s = os.environ.get(MODELSCOPE_DOMAIN)
        if s is not None and s.strip() != '':
            endpoint = MODELSCOPE_URL_SCHEME + s
            try:
                self.repo_exists(repo_id=repo_id, repo_type=repo_type, endpoint=endpoint, re_raise=True)
            except Exception:
                logger.error(f'Repo {repo_id} does not exist on {endpoint}.')
                raise
            return endpoint

        check_cn_first = not is_env_true(MODELSCOPE_PREFER_AI_SITE)
        prefer_endpoint = get_endpoint(cn_site=check_cn_first)
        if not self.repo_exists(
                repo_id, repo_type=repo_type, endpoint=prefer_endpoint):
            alternative_endpoint = get_endpoint(cn_site=(not check_cn_first))
            logger.warning(f'Repo {repo_id} not exists on {prefer_endpoint}, '
                           f'will try on alternative endpoint {alternative_endpoint}.')
            try:
                self.repo_exists(
                    repo_id, repo_type=repo_type, endpoint=alternative_endpoint, re_raise=True)
            except Exception:
                logger.error(f'Repo {repo_id} not exists on either {prefer_endpoint} or {alternative_endpoint}')
                raise
            else:
                return alternative_endpoint
        else:
            return prefer_endpoint

    def model_info(self,
                   repo_id: str,
                   *,
                   revision: Optional[str] = DEFAULT_MODEL_REVISION,
                   endpoint: Optional[str] = None) -> ModelInfo:
        """Get model information including commit history.

        Args:
            repo_id (str): The model id in the format of
                ``namespace/model_name``.
            revision (str, optional): Specific revision of the model.
                Defaults to ``DEFAULT_MODEL_REVISION``.
            endpoint (str, optional): Hub endpoint to use. When ``None``,
                use the endpoint specified when initializing :class:`HubApi`.

        Returns:
            ModelInfo: The model detailed information returned by
            ModelScope Hub with commit history.
        """
        owner_or_group, _ = model_id_to_group_owner_name(repo_id)
        model_data = self.get_model(
            model_id=repo_id, revision=revision, endpoint=endpoint)
        commits = self.list_repo_commits(
            repo_id=repo_id, repo_type=REPO_TYPE_MODEL, revision=revision, endpoint=endpoint)
        siblings = self.get_model_files(
            model_id=repo_id, revision=revision, recursive=True, endpoint=endpoint)

        # Create ModelInfo from API response data
        model_info = ModelInfo(**model_data, commits=commits, author=owner_or_group, siblings=siblings)

        return model_info

    def dataset_info(self,
                     repo_id: str,
                     *,
                     revision: Optional[str] = None,
                     endpoint: Optional[str] = None) -> DatasetInfo:
        """Get dataset information including commit history.

        Args:
            repo_id (str): The dataset id in the format of
                ``namespace/dataset_name``.
            revision (str, optional): Specific revision of the dataset.
                Defaults to ``None``.
            endpoint (str, optional): Hub endpoint to use. When ``None``,
                use the endpoint specified when initializing :class:`HubApi`.

        Returns:
            DatasetInfo: The dataset detailed information returned by
            ModelScope Hub with commit history.
        """
        owner_or_group, _ = model_id_to_group_owner_name(repo_id)
        dataset_data = self.get_dataset(
            dataset_id=repo_id, revision=revision, endpoint=endpoint)
        commits = self.list_repo_commits(
            repo_id=repo_id, repo_type=REPO_TYPE_DATASET, revision=revision, endpoint=endpoint)
        siblings = self.get_dataset_files(
            repo_id=repo_id, revision=revision or DEFAULT_DATASET_REVISION, recursive=True, endpoint=endpoint)

        # Create DatasetInfo from API response data
        dataset_info = DatasetInfo(**dataset_data, commits=commits, author=owner_or_group, siblings=siblings)

        return dataset_info

    def repo_info(
        self,
        repo_id: str,
        *,
        repo_type: Optional[str] = REPO_TYPE_MODEL,
        revision: Optional[str] = DEFAULT_MODEL_REVISION,
        endpoint: Optional[str] = None
    ) -> Union[ModelInfo, DatasetInfo]:
        """Get repository information for models or datasets.

        Args:
            repo_id (str): The repository id in the format of
                ``namespace/repo_name``.
            revision (str, optional): Specific revision of the repository.
                Currently only effective for model repositories. Defaults to
                ``DEFAULT_MODEL_REVISION``.
            repo_type (str, optional): Type of the repository. Supported
                values are ``"model"`` and ``"dataset"``. If not provided,
                ``"model"`` is assumed.
            endpoint (str, optional): Hub endpoint to use. When ``None``,
                use the endpoint specified when initializing :class:`HubApi`.

        Returns:
            Union[ModelInfo, DatasetInfo]: The repository detailed information
            returned by ModelScope Hub.
        """
        if repo_type is None or repo_type == REPO_TYPE_MODEL:
            return self.model_info(repo_id=repo_id, revision=revision, endpoint=endpoint)

        if repo_type == REPO_TYPE_DATASET:
            return self.dataset_info(repo_id=repo_id, revision=revision, endpoint=endpoint)

        raise InvalidParameter(
            f'Arg repo_type {repo_type} not supported. Please choose from {REPO_TYPE_SUPPORT}.')

    def repo_exists(
            self,
            repo_id: str,
            *,
            repo_type: Optional[str] = None,
            endpoint: Optional[str] = None,
            re_raise: Optional[bool] = False,
            token: Optional[str] = None
    ) -> bool:
        """
        Checks if a repository exists on ModelScope

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            repo_type (`str`, *optional*):
                `None` or `"model"` if getting repository info from a model. Default is `None`.
                TODO: support studio
            endpoint(`str`):
                None or specific endpoint to use, when None, use the default endpoint
                set in HubApi class (self.endpoint)
            re_raise(`bool`):
                raise exception when error
            token (`str`, *optional*): access token to use for checking existence.
        Returns:
            True if the repository exists, False otherwise.
        """
        if endpoint is None:
            endpoint = self.endpoint
        if (repo_type is not None) and repo_type.lower() not in REPO_TYPE_SUPPORT:
            raise Exception('Not support repo-type: %s' % repo_type)
        if (repo_id is None) or repo_id.count('/') != 1:
            raise Exception('Invalid repo_id: %s, must be of format namespace/name' % repo_type)

        cookies = self.get_cookies(access_token=token, cookies_required=False)
        owner_or_group, name = model_id_to_group_owner_name(repo_id)
        if (repo_type is not None) and repo_type.lower() == REPO_TYPE_DATASET:
            path = f'{endpoint}/api/v1/datasets/{owner_or_group}/{name}'
        else:
            path = f'{endpoint}/api/v1/models/{owner_or_group}/{name}'

        r = self.session.get(path, cookies=cookies,
                             headers=self.builder_headers(self.headers))
        code = handle_http_response(r, logger, cookies, repo_id, False)
        if code == 200:
            return True
        elif code == 404:
            if re_raise:
                raise HTTPError(r)
            else:
                return False
        else:
            logger.warn(f'Check repo_exists return status code {code}.')
            raise Exception(
                'Failed to check existence of repo: %s, make sure you have access authorization.'
                % repo_type)

    def delete_repo(self, repo_id: str, repo_type: str, endpoint: Optional[str] = None):
        """
        Delete a repository from ModelScope.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            repo_type (`str`):
                The type of the repository. Supported types are `model` and `dataset`.
            endpoint(`str`):
                The endpoint to use. If not provided, the default endpoint is `https://www.modelscope.cn`
                Could be set to `https://ai.modelscope.ai` for international version.
        """

        if not endpoint:
            endpoint = self.endpoint

        if repo_type == REPO_TYPE_DATASET:
            self.delete_dataset(repo_id, endpoint)
        elif repo_type == REPO_TYPE_MODEL:
            self.delete_model(repo_id, endpoint)
        else:
            raise Exception(f'Arg repo_type {repo_type} not supported.')

        logger.info(f'Repo {repo_id} deleted successfully.')

    @staticmethod
    def _create_default_config(model_dir):
        cfg_file = os.path.join(model_dir, ModelFile.CONFIGURATION)
        cfg = {
            ConfigFields.framework: Frameworks.torch,
            ConfigFields.task: Tasks.other,
        }
        with open(cfg_file, 'w') as file:
            json.dump(cfg, file)

    def push_model(self,
                   model_id: str,
                   model_dir: str,
                   visibility: Optional[int] = ModelVisibility.PUBLIC,
                   license: Optional[str] = Licenses.APACHE_V2,
                   chinese_name: Optional[str] = None,
                   commit_message: Optional[str] = 'upload model',
                   tag: Optional[str] = None,
                   revision: Optional[str] = DEFAULT_REPOSITORY_REVISION,
                   original_model_id: Optional[str] = None,
                   ignore_file_pattern: Optional[Union[List[str], str]] = None,
                   lfs_suffix: Optional[Union[str, List[str]]] = None):
        warnings.warn(
            'This function is deprecated and will be removed in future versions. '
            'Please use git command directly or use HubApi().upload_folder instead',
            DeprecationWarning,
            stacklevel=2
        )
        """Upload model from a given directory to given repository. A valid model directory
        must contain a configuration.json file.

        This function upload the files in given directory to given repository. If the
        given repository is not exists in remote, it will automatically create it with
        given visibility, license and chinese_name parameters. If the revision is also
        not exists in remote repository, it will create a new branch for it.

        This function must be called before calling HubApi's login with a valid token
        which can be obtained from ModelScope's website.

        If any error, please upload via git commands.

        Args:
            model_id (str):
                The model id to be uploaded, caller must have write permission for it.
            model_dir(str):
                The Absolute Path of the finetune result.
            visibility(int, optional):
                Visibility of the new created model(1-private, 5-public). If the model is
                not exists in ModelScope, this function will create a new model with this
                visibility and this parameter is required. You can ignore this parameter
                if you make sure the model's existence.
            license(`str`, defaults to `None`):
                License of the new created model(see License). If the model is not exists
                in ModelScope, this function will create a new model with this license
                and this parameter is required. You can ignore this parameter if you
                make sure the model's existence.
            chinese_name(`str`, *optional*, defaults to `None`):
                chinese name of the new created model.
            commit_message(`str`, *optional*, defaults to `None`):
                commit message of the push request.
            tag(`str`, *optional*, defaults to `None`):
                The tag on this commit
            revision (`str`, *optional*, default to DEFAULT_MODEL_REVISION):
                which branch to push. If the branch is not exists, It will create a new
                branch and push to it.
            original_model_id (str, optional): The base model id which this model is trained from
            ignore_file_pattern (`Union[List[str], str]`, optional): The file pattern to ignore uploading
            lfs_suffix (`List[str]`, optional): File types to use LFS to manage. examples: '*.safetensors'.

        Raises:
            InvalidParameter: Parameter invalid.
            NotLoginException: Not login
            ValueError: No configuration.json
            Exception: Create failed.
        """
        if model_id is None:
            raise InvalidParameter('model_id cannot be empty!')
        if model_dir is None:
            raise InvalidParameter('model_dir cannot be empty!')
        if not os.path.exists(model_dir) or os.path.isfile(model_dir):
            raise InvalidParameter('model_dir must be a valid directory.')
        cfg_file = os.path.join(model_dir, ModelFile.CONFIGURATION)
        if not os.path.exists(cfg_file):
            logger.warning(
                f'No {ModelFile.CONFIGURATION} file found in {model_dir}, creating a default one.')
            HubApi._create_default_config(model_dir)

        cookies = ModelScopeConfig.get_cookies()
        if cookies is None:
            raise NotLoginException('Must login before upload!')
        files_to_save = os.listdir(model_dir)
        folder_size = get_readable_folder_size(model_dir)
        if ignore_file_pattern is None:
            ignore_file_pattern = []
        if isinstance(ignore_file_pattern, str):
            ignore_file_pattern = [ignore_file_pattern]
        if visibility is None or license is None:
            raise InvalidParameter('Visibility and License cannot be empty for new model.')
        if not self.repo_exists(model_id):
            logger.info('Creating new model [%s]' % model_id)
            self.create_model(
                model_id=model_id,
                visibility=visibility,
                license=license,
                chinese_name=chinese_name,
                original_model_id=original_model_id)
        tmp_dir = os.path.join(model_dir, TEMPORARY_FOLDER_NAME)  # make temporary folder
        git_wrapper = GitCommandWrapper()
        logger.info(f'Pushing folder {model_dir} as model {model_id}.')
        logger.info(f'Total folder size {folder_size}, this may take a while depending on actual pushing size...')
        try:
            repo = Repository(model_dir=tmp_dir, clone_from=model_id)
            branches = git_wrapper.get_remote_branches(tmp_dir)
            if revision not in branches:
                logger.info('Creating new branch %s' % revision)
                git_wrapper.new_branch(tmp_dir, revision)
            git_wrapper.checkout(tmp_dir, revision)
            files_in_repo = os.listdir(tmp_dir)
            for f in files_in_repo:
                if f[0] != '.':
                    src = os.path.join(tmp_dir, f)
                    if os.path.isfile(src):
                        os.remove(src)
                    else:
                        shutil.rmtree(src, ignore_errors=True)
            for f in files_to_save:
                if f[0] != '.':
                    if any([re.search(pattern, f) is not None for pattern in ignore_file_pattern]):
                        continue
                    src = os.path.join(model_dir, f)
                    if os.path.isdir(src):
                        shutil.copytree(src, os.path.join(tmp_dir, f))
                    else:
                        shutil.copy(src, tmp_dir)
            if not commit_message:
                date = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
                commit_message = '[automsg] push model %s to hub at %s' % (
                    model_id, date)
            if lfs_suffix is not None:
                lfs_suffix_list = [lfs_suffix] if isinstance(lfs_suffix, str) else lfs_suffix
                for suffix in lfs_suffix_list:
                    repo.add_lfs_type(suffix)
            repo.push(
                commit_message=commit_message,
                local_branch=revision,
                remote_branch=revision)
            if tag is not None:
                repo.tag_and_push(tag, tag)
            logger.info(f'Successfully push folder {model_dir} to remote repo [{model_id}].')
        except Exception:
            raise
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def list_models(self,
                    owner_or_group: str,
                    page_number: Optional[int] = 1,
                    page_size: Optional[int] = 10,
                    endpoint: Optional[str] = None) -> dict:
        """List models in owner or group.

        Args:
            owner_or_group(str): owner or group.
            page_number(int, optional): The page number, default: 1
            page_size(int, optional): The page size, default: 10
            endpoint: the endpoint to use, default to None to use endpoint specified in the class

        Raises:
            RequestError: The request error.

        Returns:
            dict: {"models": "list of models", "TotalCount": total_number_of_models_in_owner_or_group}
        """
        cookies = ModelScopeConfig.get_cookies()
        if not endpoint:
            endpoint = self.endpoint
        path = f'{endpoint}/api/v1/models/'
        r = self.session.put(
            path,
            data='{"Path":"%s", "PageNumber":%s, "PageSize": %s}' %
                 (owner_or_group, page_number, page_size),
            cookies=cookies,
            headers=self.builder_headers(self.headers))
        handle_http_response(r, logger, cookies, owner_or_group)
        if r.status_code == HTTPStatus.OK:
            if is_ok(r.json()):
                data = r.json()[API_RESPONSE_FIELD_DATA]
                return data
            else:
                raise RequestError(r.json()[API_RESPONSE_FIELD_MESSAGE])
        else:
            raise_for_http_status(r)
        return None

    def list_datasets(self,
                      owner_or_group: str,
                      *,
                      page_number: Optional[int] = 1,
                      page_size: Optional[int] = 10,
                      sort: Optional[str] = None,
                      search: Optional[str] = None,
                      endpoint: Optional[str] = None,
                      ) -> dict:
        """List datasets via OpenAPI with pagination, filtering and sorting.

        Args:
            owner_or_group (str): Search by dataset authors (including organizations and individuals).
            page_number (int, optional): The page number. Defaults to 1.
            page_size (int, optional): The page size. Defaults to 10.
            sort (str, optional): Sort key. If not provided, the server's default sorting is used.
                choose from ['default', 'downloads', 'likes', 'last_modified'].
            search (str, optional): Search by substring keywords in the dataset's Chinese name,
                English name, and authors (including organizations and individuals).
            endpoint (str, optional): Hub endpoint to use. When None, use the endpoint specified in the class.

        Returns:
            dict: The OpenAPI data payload, e.g.
                  {
                    "datasets": [...],
                    "total_count": int,
                    "page_number": int,
                    "page_size": int
                  }
        """
        if not endpoint:
            endpoint = self.endpoint
        path = f'{endpoint}/openapi/v1/datasets'

        # Build query params
        params: Dict[str, Any] = {
            'page_number': page_number,
            'page_size': page_size,
        }
        if sort:
            if sort not in VALID_SORT_KEYS:
                raise InvalidParameter(
                    f'Invalid sort key: {sort}. Supported sort keys: {list(VALID_SORT_KEYS)}')
            params['sort'] = sort
        if search:
            params['search'] = search
        if owner_or_group:
            params['author'] = owner_or_group

        cookies = ModelScopeConfig.get_cookies()
        headers = self.builder_headers(self.headers)

        r = self.session.get(
            path,
            params=params,
            cookies=cookies,
            headers=headers
        )
        raise_for_http_status(r)
        resp = r.json()

        # OpenAPI success schema
        if resp.get('success') is True and 'data' in resp:
            return resp['data']
        else:
            # Fallback for unexpected schema
            msg = resp.get('message') or 'Failed to list datasets'
            raise RequestError(msg)

    def _check_cookie(self, use_cookies: Union[bool, CookieJar] = False) -> CookieJar:  # noqa
        cookies = None
        if isinstance(use_cookies, CookieJar):
            cookies = use_cookies
        elif isinstance(use_cookies, bool):
            cookies = ModelScopeConfig.get_cookies()
            if use_cookies and cookies is None:
                raise ValueError('Token does not exist, please login first.')
        return cookies

    def list_model_revisions(
            self,
            model_id: str,
            cutoff_timestamp: Optional[int] = None,
            use_cookies: Union[bool, CookieJar] = False) -> List[str]:
        """Get model branch and tags.

        Args:
            model_id (str): The model id
            cutoff_timestamp (int): Tags created before the cutoff will be included.
                                    The timestamp is represented by the seconds elapsed from the epoch time.
            use_cookies (Union[bool, CookieJar], optional): If is cookieJar, we will use this cookie, if True,
                        will load cookie from local. Defaults to False.

        Returns:
            Tuple[List[str], List[str]]: Return list of branch name and tags
        """
        tags_details = self.list_model_revisions_detail(model_id=model_id,
                                                        cutoff_timestamp=cutoff_timestamp,
                                                        use_cookies=use_cookies)
        tags = [x['Revision'] for x in tags_details
                ] if tags_details else []
        return tags

    def list_model_revisions_detail(
            self,
            model_id: str,
            cutoff_timestamp: Optional[int] = None,
            use_cookies: Union[bool, CookieJar] = False,
            endpoint: Optional[str] = None) -> List[str]:
        """Get model branch and tags.

        Args:
            model_id (str): The model id
            cutoff_timestamp (int): Tags created before the cutoff will be included.
                                    The timestamp is represented by the seconds elapsed from the epoch time.
            use_cookies (Union[bool, CookieJar], optional): If is cookieJar, we will use this cookie, if True,
                        will load cookie from local. Defaults to False.
            endpoint: the endpoint to use, default to None to use endpoint specified in the class

        Returns:
            Tuple[List[str], List[str]]: Return list of branch name and tags
        """
        cookies = self._check_cookie(use_cookies)
        if cutoff_timestamp is None:
            cutoff_timestamp = get_release_datetime()
        if not endpoint:
            endpoint = self.endpoint
        path = f'{endpoint}/api/v1/models/{model_id}/revisions?EndTime=%s' % cutoff_timestamp
        r = self.session.get(path, cookies=cookies,
                             headers=self.builder_headers(self.headers))
        handle_http_response(r, logger, cookies, model_id)
        d = r.json()
        raise_on_error(d)
        info = d[API_RESPONSE_FIELD_DATA]
        # tags returned from backend are guaranteed to be ordered by create-time
        return info['RevisionMap']['Tags']

    def get_branch_tag_detail(self, details, name):
        for item in details:
            if item['Revision'] == name:
                return item
        return None

    def get_valid_revision_detail(self,
                                  model_id: str,
                                  revision=None,
                                  cookies: Optional[CookieJar] = None,
                                  endpoint: Optional[str] = None):
        if not endpoint:
            endpoint = self.endpoint
        release_timestamp = get_release_datetime()
        current_timestamp = int(round(datetime.datetime.now().timestamp()))
        # for active development in library codes (non-release-branches), release_timestamp
        # is set to be a far-away-time-in-the-future, to ensure that we shall
        # get the master-HEAD version from model repo by default (when no revision is provided)
        all_branches_detail, all_tags_detail = self.get_model_branches_and_tags_details(
            model_id, use_cookies=False if cookies is None else cookies, endpoint=endpoint)
        all_branches = [x['Revision'] for x in all_branches_detail] if all_branches_detail else []
        all_tags = [x['Revision'] for x in all_tags_detail] if all_tags_detail else []
        if release_timestamp > current_timestamp + ONE_YEAR_SECONDS:
            if revision is None:
                revision = MASTER_MODEL_BRANCH
                logger.info(
                    'Model revision not specified, using default [%s] version.'
                    % revision)
            if revision not in all_branches and revision not in all_tags:
                raise NotExistError('The model: %s has no revision : %s .' % (model_id, revision))

            revision_detail = self.get_branch_tag_detail(all_tags_detail, revision)
            if revision_detail is None:
                revision_detail = self.get_branch_tag_detail(all_branches_detail, revision)
            logger.debug('Development mode use revision: %s' % revision)
        else:
            if revision is not None and revision in all_branches:
                revision_detail = self.get_branch_tag_detail(all_branches_detail, revision)
                return revision_detail

            if len(all_tags_detail) == 0:  # use no revision use master as default.
                if revision is None or revision == MASTER_MODEL_BRANCH:
                    revision = MASTER_MODEL_BRANCH
                else:
                    raise NotExistError('The model: %s has no revision: %s !' % (model_id, revision))
                revision_detail = self.get_branch_tag_detail(all_branches_detail, revision)
            else:
                if revision is None:  # user not specified revision, use latest revision before release time
                    revisions_detail = [x for x in
                                        all_tags_detail if
                                        x['CreatedAt'] <= release_timestamp] if all_tags_detail else []  # noqa E501
                    if len(revisions_detail) > 0:
                        revision = revisions_detail[0]['Revision']  # use latest revision before release time.
                        revision_detail = revisions_detail[0]
                    else:
                        revision = MASTER_MODEL_BRANCH
                        revision_detail = self.get_branch_tag_detail(all_branches_detail, revision)
                        vl = '[%s]' % ','.join(all_tags)
                        logger.warning('Model revision should be specified from revisions: %s' % (vl))
                    logger.warning('Model revision not specified, use revision: %s' % revision)
                else:
                    # use user-specified revision
                    if revision not in all_tags:
                        if revision == MASTER_MODEL_BRANCH:
                            logger.warning('Using the master branch is fragile, please use it with caution!')
                            revision_detail = self.get_branch_tag_detail(all_branches_detail, revision)
                        else:
                            vl = '[%s]' % ','.join(all_tags)
                            raise NotExistError('The model: %s has no revision: %s valid are: %s!' %
                                                (model_id, revision, vl))
                    else:
                        revision_detail = self.get_branch_tag_detail(all_tags_detail, revision)
                    logger.info('Use user-specified model revision: %s' % revision)
        return revision_detail

    def get_valid_revision(self,
                           model_id: str,
                           revision=None,
                           cookies: Optional[CookieJar] = None,
                           endpoint: Optional[str] = None):
        return self.get_valid_revision_detail(model_id=model_id,
                                              revision=revision,
                                              cookies=cookies,
                                              endpoint=endpoint)['Revision']

    def get_model_branches_and_tags_details(
            self,
            model_id: str,
            use_cookies: Union[bool, CookieJar] = False,
            endpoint: Optional[str] = None
    ) -> Tuple[List[str], List[str]]:
        """Get model branch and tags.

        Args:
            model_id (str): The model id
            use_cookies (Union[bool, CookieJar], optional): If is cookieJar, we will use this cookie, if True,
                        will load cookie from local. Defaults to False.
            endpoint: the endpoint to use, default to None to use endpoint specified in the class

        Returns:
            Tuple[List[str], List[str]]: Return list of branch name and tags
        """
        cookies = self._check_cookie(use_cookies)
        if not endpoint:
            endpoint = self.endpoint
        path = f'{endpoint}/api/v1/models/{model_id}/revisions'
        r = self.session.get(path, cookies=cookies,
                             headers=self.builder_headers(self.headers))
        handle_http_response(r, logger, cookies, model_id)
        d = r.json()
        raise_on_error(d)
        info = d[API_RESPONSE_FIELD_DATA]
        return info['RevisionMap']['Branches'], info['RevisionMap']['Tags']

    def get_model_branches_and_tags(
            self,
            model_id: str,
            use_cookies: Union[bool, CookieJar] = False,
    ) -> Tuple[List[str], List[str]]:
        """Get model branch and tags.

        Args:
            model_id (str): The model id
            use_cookies (Union[bool, CookieJar], optional): If is cookieJar, we will use this cookie, if True,
                        will load cookie from local. Defaults to False.

        Returns:
            Tuple[List[str], List[str]]: Return list of branch name and tags
        """
        branches_detail, tags_detail = self.get_model_branches_and_tags_details(model_id=model_id,
                                                                                use_cookies=use_cookies)
        branches = [x['Revision'] for x in branches_detail
                    ] if branches_detail else []
        tags = [x['Revision'] for x in tags_detail
                ] if tags_detail else []
        return branches, tags

    def get_model_files(self,
                        model_id: str,
                        revision: Optional[str] = DEFAULT_MODEL_REVISION,
                        root: Optional[str] = None,
                        recursive: Optional[bool] = False,
                        use_cookies: Union[bool, CookieJar] = False,
                        headers: Optional[dict] = {},
                        endpoint: Optional[str] = None) -> List[dict]:
        """List the models files.

        Args:
            model_id (str): The model id
            revision (Optional[str], optional): The branch or tag name.
            root (Optional[str], optional): The root path. Defaults to None.
            recursive (Optional[bool], optional): Is recursive list files. Defaults to False.
            use_cookies (Union[bool, CookieJar], optional): If is cookieJar, we will use this cookie, if True,
                        will load cookie from local. Defaults to False.
            headers: request headers
            endpoint: the endpoint to use, default to None to use endpoint specified in the class

        Returns:
            List[dict]: Model file list.
        """
        if not endpoint:
            endpoint = self.endpoint
        if revision:
            path = '%s/api/v1/models/%s/repo/files?Revision=%s&Recursive=%s' % (
                endpoint, model_id, revision, recursive)
        else:
            path = '%s/api/v1/models/%s/repo/files?Recursive=%s' % (
                endpoint, model_id, recursive)
        cookies = self._check_cookie(use_cookies)
        if root is not None:
            path = path + f'&Root={root}'
        headers = self.headers if headers is None else headers
        headers['X-Request-ID'] = str(uuid.uuid4().hex)
        r = self.session.get(
            path, cookies=cookies, headers=headers)

        handle_http_response(r, logger, cookies, model_id)
        d = r.json()
        raise_on_error(d)

        files = []
        if not d[API_RESPONSE_FIELD_DATA]['Files']:
            logger.warning(f'No files found in model {model_id} at revision {revision}.')
            return files
        for file in d[API_RESPONSE_FIELD_DATA]['Files']:
            if file['Name'] == '.gitignore' or file['Name'] == '.gitattributes':
                continue

            files.append(file)
        return files

    def file_exists(
            self,
            repo_id: str,
            filename: str,
            *,
            revision: Optional[str] = None,
    ):
        """Get if the specified file exists

        Args:
            repo_id (`str`): The repo id to use
            filename (`str`): The queried filename, if the file exists in a sub folder,
                please pass <sub-folder-name>/<file-name>
            revision (`Optional[str]`): The repo revision
        Returns:
            The query result in bool value
        """
        cookies = ModelScopeConfig.get_cookies()
        files = self.get_model_files(
            repo_id,
            recursive=True,
            revision=revision,
            use_cookies=False if cookies is None else cookies,
        )
        files = [file['Path'] for file in files]
        return filename in files

    def create_dataset(self,
                       dataset_name: str,
                       namespace: str,
                       chinese_name: Optional[str] = '',
                       license: Optional[str] = Licenses.APACHE_V2,
                       visibility: Optional[int] = DatasetVisibility.PUBLIC,
                       description: Optional[str] = '',
                       endpoint: Optional[str] = None, ) -> str:

        if dataset_name is None or namespace is None:
            raise InvalidParameter('dataset_name and namespace are required!')

        cookies = ModelScopeConfig.get_cookies()
        if cookies is None:
            raise ValueError('Token does not exist, please login first.')
        if not endpoint:
            endpoint = self.endpoint
        path = f'{endpoint}/api/v1/datasets'
        files = {
            'Name': (None, dataset_name),
            'ChineseName': (None, chinese_name),
            'Owner': (None, namespace),
            'License': (None, license),
            'Visibility': (None, visibility),
            'Description': (None, description)
        }

        r = self.session.post(
            path,
            files=files,
            cookies=cookies,
            headers=self.builder_headers(self.headers),
        )

        handle_http_post_error(r, path, files)
        raise_on_error(r.json())
        dataset_repo_url = f'{endpoint}/datasets/{namespace}/{dataset_name}'
        logger.info(f'Create dataset success: {dataset_repo_url}')
        return dataset_repo_url

    def delete_dataset(self, dataset_id: str, endpoint: Optional[str] = None):

        cookies = ModelScopeConfig.get_cookies()
        if not endpoint:
            endpoint = self.endpoint
        if cookies is None:
            raise ValueError('Token does not exist, please login first.')

        path = f'{endpoint}/api/v1/datasets/{dataset_id}'
        r = self.session.delete(path,
                                cookies=cookies,
                                headers=self.builder_headers(self.headers))
        raise_for_http_status(r)
        raise_on_error(r.json())

    def get_dataset_id_and_type(self, dataset_name: str, namespace: str, endpoint: Optional[str] = None):
        """ Get the dataset id and type. """
        if not endpoint:
            endpoint = self.endpoint
        datahub_url = f'{endpoint}/api/v1/datasets/{namespace}/{dataset_name}'
        cookies = ModelScopeConfig.get_cookies()
        r = self.session.get(datahub_url, cookies=cookies)
        resp = r.json()
        datahub_raise_on_error(datahub_url, resp, r)
        dataset_id = resp['Data']['Id']
        dataset_type = resp['Data']['Type']
        return dataset_id, dataset_type

    def list_repo_tree(self,
                       dataset_name: str,
                       namespace: str,
                       revision: str,
                       root_path: str,
                       recursive: bool = True,
                       page_number: int = 1,
                       page_size: int = 100,
                       endpoint: Optional[str] = None):
        """
        @deprecated: Use `get_dataset_files` instead.
        """
        warnings.warn('The function `list_repo_tree` is deprecated, use `get_dataset_files` instead.',
                      DeprecationWarning)

        dataset_hub_id, dataset_type = self.get_dataset_id_and_type(
            dataset_name=dataset_name, namespace=namespace, endpoint=endpoint)

        recursive = 'True' if recursive else 'False'
        if not endpoint:
            endpoint = self.endpoint
        datahub_url = f'{endpoint}/api/v1/datasets/{dataset_hub_id}/repo/tree'
        params = {'Revision': revision if revision else 'master',
                  'Root': root_path if root_path else '/', 'Recursive': recursive,
                  'PageNumber': page_number, 'PageSize': page_size}
        cookies = ModelScopeConfig.get_cookies()

        r = self.session.get(datahub_url, params=params, cookies=cookies)
        resp = r.json()
        datahub_raise_on_error(datahub_url, resp, r)

        return resp

    def list_repo_commits(self,
                          repo_id: str,
                          *,
                          repo_type: Optional[str] = REPO_TYPE_MODEL,
                          revision: Optional[str] = DEFAULT_REPOSITORY_REVISION,
                          page_number: int = 1,
                          page_size: int = 50,
                          endpoint: Optional[str] = None):
        """
        Get the commit history for a repository.

        Args:
            repo_id (str): The repository id, in the format of `namespace/repo_name`.
            repo_type (Optional[str]): The type of the repository. Supported types are `model` and `dataset`.
            revision (str): The branch or tag name. Defaults to `DEFAULT_REPOSITORY_REVISION`.
            page_number (int): The page number for pagination. Defaults to 1.
            page_size (int): The number of commits per page. Defaults to 50.
            endpoint (Optional[str]): The endpoint to use, defaults to None to use the endpoint specified in the class.

        Returns:
            CommitHistoryResponse: The commit history response.

        Examples:
            >>> from modelscope.hub.api import HubApi
            >>> api = HubApi()
            >>> commit_history = api.list_repo_commits('meituan/Meeseeks')
            >>> print(f"Total commits: {commit_history.total_count}")
            >>> for commit in commit_history.commits:
            ...     print(f"{commit.short_id}: {commit.title}")
        """
        from datasets.utils.file_utils import is_relative_path

        if is_relative_path(repo_id) and repo_id.count('/') == 1:
            _owner, _dataset_name = repo_id.split('/')
        else:
            raise ValueError(f'Invalid repo_id: {repo_id} !')

        if not endpoint:
            endpoint = self.endpoint

        commits_url = f'{endpoint}/api/v1/{repo_type}s/{repo_id}/commits' if repo_type else \
            f'{endpoint}/api/v1/models/{repo_id}/commits'
        params = {
            'Ref': revision or DEFAULT_REPOSITORY_REVISION,
            'PageNumber': page_number,
            'PageSize': page_size
        }
        cookies = ModelScopeConfig.get_cookies()

        try:
            r = self.session.get(commits_url, params=params,
                                 cookies=cookies, headers=self.builder_headers(self.headers))
            raise_for_http_status(r)
            resp = r.json()
            raise_on_error(resp)

            if resp.get('Code') == HTTPStatus.OK:
                return CommitHistoryResponse.from_api_response(resp)

        except requests.exceptions.RequestException as e:
            raise Exception(f'Failed to get repository commits for {repo_id}: {str(e)}')

    def get_dataset_files(self,
                          repo_id: str,
                          *,
                          revision: str = DEFAULT_REPOSITORY_REVISION,
                          root_path: str = '/',
                          recursive: bool = True,
                          page_number: int = 1,
                          page_size: int = 100,
                          endpoint: Optional[str] = None):
        """
        Get the dataset files.

        Args:
            repo_id (str): The repository id, in the format of `namespace/dataset_name`.
            revision (str): The branch or tag name. Defaults to `DEFAULT_REPOSITORY_REVISION`.
            root_path (str): The root path to list. Defaults to '/'.
            recursive (bool): Whether to list recursively. Defaults to True.
            page_number (int): The page number for pagination. Defaults to 1.
            page_size (int): The number of items per page. Defaults to 100.
            endpoint (Optional[str]): The endpoint to use, defaults to None to use the endpoint specified in the class.

        Returns:
            List: The response containing the dataset repository tree information.
                e.g. [{'CommitId': None, 'CommitMessage': '...', 'Size': 0, 'Type': 'tree'}, ...]
        """
        from datasets.utils.file_utils import is_relative_path

        if is_relative_path(repo_id) and repo_id.count('/') == 1:
            _owner, _dataset_name = repo_id.split('/')
        else:
            raise ValueError(f'Invalid repo_id: {repo_id} !')

        dataset_hub_id, dataset_type = self.get_dataset_id_and_type(
            dataset_name=_dataset_name, namespace=_owner, endpoint=endpoint)

        if not endpoint:
            endpoint = self.endpoint
        datahub_url = f'{endpoint}/api/v1/datasets/{dataset_hub_id}/repo/tree'
        params = {
            'Revision': revision,
            'Root': root_path,
            'Recursive': 'True' if recursive else 'False',
            'PageNumber': page_number,
            'PageSize': page_size
        }
        cookies = ModelScopeConfig.get_cookies()

        r = self.session.get(datahub_url, params=params, cookies=cookies)
        resp = r.json()
        datahub_raise_on_error(datahub_url, resp, r)

        return resp['Data']['Files']

    def get_dataset(
        self,
        dataset_id: str,
        revision: Optional[str] = DEFAULT_REPOSITORY_REVISION,
        endpoint: Optional[str] = None
    ):
        """
        Get the dataset information.

        Args:
            dataset_id (str): The dataset id.
            revision (Optional[str]): The revision of the dataset.
            endpoint (Optional[str]): The endpoint to use, defaults to None to use the endpoint specified in the class.

        Returns:
            dict: The dataset information.
        """
        cookies = ModelScopeConfig.get_cookies()
        if not endpoint:
            endpoint = self.endpoint

        if revision:
            path = f'{endpoint}/api/v1/datasets/{dataset_id}?Revision={revision}'
        else:
            path = f'{endpoint}/api/v1/datasets/{dataset_id}'

        r = self.session.get(
            path, cookies=cookies, headers=self.builder_headers(self.headers))
        raise_for_http_status(r)
        resp = r.json()
        datahub_raise_on_error(path, resp, r)
        return resp[API_RESPONSE_FIELD_DATA]

    def get_dataset_meta_file_list(self, dataset_name: str, namespace: str,
                                   dataset_id: str, revision: str, endpoint: Optional[str] = None):
        """ Get the meta file-list of the dataset. """
        if not endpoint:
            endpoint = self.endpoint
        datahub_url = f'{endpoint}/api/v1/datasets/{dataset_id}/repo/tree?Revision={revision}'
        cookies = ModelScopeConfig.get_cookies()
        r = self.session.get(datahub_url,
                             cookies=cookies,
                             headers=self.builder_headers(self.headers))
        resp = r.json()
        datahub_raise_on_error(datahub_url, resp, r)
        file_list = resp['Data']
        if file_list is None:
            raise NotExistError(
                f'The modelscope dataset [dataset_name = {dataset_name}, namespace = {namespace}, '
                f'version = {revision}] dose not exist')

        file_list = file_list['Files']
        return file_list

    @staticmethod
    def dump_datatype_file(dataset_type: int, meta_cache_dir: str):
        """
        Dump the data_type as a local file, in order to get the dataset
         formation without calling the datahub.
        More details, please refer to the class
        `modelscope.utils.constant.DatasetFormations`.
        """
        dataset_type_file_path = os.path.join(meta_cache_dir,
                                              f'{str(dataset_type)}{DatasetFormations.formation_mark_ext.value}')
        with open(dataset_type_file_path, 'w') as fp:
            fp.write('*** Automatically-generated file, do not modify ***')

    def get_dataset_meta_files_local_paths(self, dataset_name: str,
                                           namespace: str,
                                           revision: str,
                                           meta_cache_dir: str, dataset_type: int, file_list: list,
                                           endpoint: Optional[str] = None):
        local_paths = defaultdict(list)
        dataset_formation = DatasetFormations(dataset_type)
        dataset_meta_format = DatasetMetaFormats[dataset_formation]
        cookies = ModelScopeConfig.get_cookies()

        # Dump the data_type as a local file
        HubApi.dump_datatype_file(dataset_type=dataset_type, meta_cache_dir=meta_cache_dir)
        if not endpoint:
            endpoint = self.endpoint
        for file_info in file_list:
            file_path = file_info['Path']
            extension = os.path.splitext(file_path)[-1]
            if extension in dataset_meta_format:
                datahub_url = f'{endpoint}/api/v1/datasets/{namespace}/{dataset_name}/repo?' \
                              f'Revision={revision}&FilePath={file_path}'
                r = self.session.get(datahub_url, cookies=cookies)
                raise_for_http_status(r)
                local_path = os.path.join(meta_cache_dir, file_path)
                if os.path.exists(local_path):
                    logger.warning(
                        f"Reusing dataset {dataset_name}'s python file ({local_path})"
                    )
                    local_paths[extension].append(local_path)
                    continue
                with open(local_path, 'wb') as f:
                    f.write(r.content)
                local_paths[extension].append(local_path)

        return local_paths, dataset_formation

    @staticmethod
    def fetch_meta_files_from_url(url, out_path, chunk_size=1024, mode=DownloadMode.REUSE_DATASET_IF_EXISTS):
        """
        Fetch the meta-data files from the url, e.g. csv/jsonl files.
        """
        import hashlib
        from tqdm.auto import tqdm
        import pandas as pd

        out_path = os.path.join(out_path, hashlib.md5(url.encode(encoding='UTF-8')).hexdigest())
        if mode == DownloadMode.FORCE_REDOWNLOAD and os.path.exists(out_path):
            os.remove(out_path)
        if os.path.exists(out_path):
            logger.info(f'Reusing cached meta-data file: {out_path}')
            return out_path
        cookies = ModelScopeConfig.get_cookies()

        # Make the request and get the response content as TextIO
        logger.info('Loading meta-data file ...')
        response = requests.get(url, cookies=cookies, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        progress = tqdm(total=total_size, dynamic_ncols=True)

        def get_chunk(resp):
            chunk_data = []
            for data in resp.iter_lines():
                data = data.decode('utf-8')
                chunk_data.append(data)
                if len(chunk_data) >= chunk_size:
                    yield chunk_data
                    chunk_data = []
            yield chunk_data

        iter_num = 0
        with open(out_path, 'a') as f:
            for chunk in get_chunk(response):
                progress.update(len(chunk))
                if url.endswith('jsonl'):
                    chunk = [json.loads(line) for line in chunk if line.strip()]
                    if len(chunk) == 0:
                        continue
                    if iter_num == 0:
                        with_header = True
                    else:
                        with_header = False
                    chunk_df = pd.DataFrame(chunk)
                    chunk_df.to_csv(f, index=False, header=with_header, escapechar='\\')
                    iter_num += 1
                else:
                    # csv or others
                    for line in chunk:
                        f.write(line + '\n')
        progress.close()

        return out_path

    def get_dataset_file_url(
            self,
            file_name: str,
            dataset_name: str,
            namespace: str,
            revision: Optional[str] = DEFAULT_DATASET_REVISION,
            view: Optional[bool] = False,
            extension_filter: Optional[bool] = True,
            endpoint: Optional[str] = None):

        if not file_name or not dataset_name or not namespace:
            raise ValueError('Args (file_name, dataset_name, namespace) cannot be empty!')

        # Note: make sure the FilePath is the last parameter in the url
        params: dict = {'Source': 'SDK', 'Revision': revision, 'FilePath': file_name, 'View': view}
        params: str = urlencode(params)
        if not endpoint:
            endpoint = self.endpoint
        file_url = f'{endpoint}/api/v1/datasets/{namespace}/{dataset_name}/repo?{params}'

        return file_url

        # if extension_filter:
        #     if os.path.splitext(file_name)[-1] in META_FILES_FORMAT:
        #         file_url = f'{self.endpoint}/api/v1/datasets/{namespace}/{dataset_name}/repo?'\
        #                    f'Revision={revision}&FilePath={file_name}'
        #     else:
        #         file_url = file_name
        #     return file_url
        # else:
        #     return file_url

    def get_dataset_file_url_origin(
            self,
            file_name: str,
            dataset_name: str,
            namespace: str,
            revision: Optional[str] = DEFAULT_DATASET_REVISION,
            endpoint: Optional[str] = None):
        if not endpoint:
            endpoint = self.endpoint
        if file_name and os.path.splitext(file_name)[-1] in META_FILES_FORMAT:
            file_name = f'{endpoint}/api/v1/datasets/{namespace}/{dataset_name}/repo?' \
                        f'Revision={revision}&FilePath={file_name}'
        return file_name

    def get_dataset_access_config(
            self,
            dataset_name: str,
            namespace: str,
            revision: Optional[str] = DEFAULT_DATASET_REVISION,
            endpoint: Optional[str] = None):
        if not endpoint:
            endpoint = self.endpoint
        datahub_url = f'{endpoint}/api/v1/datasets/{namespace}/{dataset_name}/' \
                      f'ststoken?Revision={revision}'
        return self.datahub_remote_call(datahub_url)

    def get_dataset_access_config_session(
            self,
            dataset_name: str,
            namespace: str,
            check_cookie: bool,
            revision: Optional[str] = DEFAULT_DATASET_REVISION,
            endpoint: Optional[str] = None):

        if not endpoint:
            endpoint = self.endpoint
        datahub_url = f'{endpoint}/api/v1/datasets/{namespace}/{dataset_name}/' \
                      f'ststoken?Revision={revision}'
        if check_cookie:
            cookies = self._check_cookie(use_cookies=True)
        else:
            cookies = ModelScopeConfig.get_cookies()

        r = self.session.get(
            url=datahub_url,
            cookies=cookies,
            headers=self.builder_headers(self.headers))
        resp = r.json()
        raise_on_error(resp)
        return resp['Data']

    def get_virgo_meta(self, dataset_id: str, version: int = 1) -> dict:
        """
        Get virgo dataset meta info.
        """
        virgo_endpoint = os.environ.get(VirgoDatasetConfig.env_virgo_endpoint, '')
        if not virgo_endpoint:
            raise RuntimeError(f'Virgo endpoint is not set in env: {VirgoDatasetConfig.env_virgo_endpoint}')

        virgo_dataset_url = f'{virgo_endpoint}/data/set/download'
        cookies = requests.utils.dict_from_cookiejar(ModelScopeConfig.get_cookies())

        dataset_info = dict(
            dataSetId=dataset_id,
            dataSetVersion=version
        )
        data = dict(
            data=dataset_info,
        )
        r = self.session.post(url=virgo_dataset_url,
                              json=data,
                              cookies=cookies,
                              headers=self.builder_headers(self.headers),
                              timeout=900)
        resp = r.json()
        if resp['code'] != 0:
            raise RuntimeError(f'Failed to get virgo dataset: {resp}')

        return resp['data']

    def get_dataset_access_config_for_unzipped(self,
                                               dataset_name: str,
                                               namespace: str,
                                               revision: str,
                                               zip_file_name: str,
                                               endpoint: Optional[str] = None):
        if not endpoint:
            endpoint = self.endpoint
        datahub_url = f'{endpoint}/api/v1/datasets/{namespace}/{dataset_name}'
        cookies = ModelScopeConfig.get_cookies()
        r = self.session.get(url=datahub_url, cookies=cookies,
                             headers=self.builder_headers(self.headers))
        resp = r.json()
        # get visibility of the dataset
        raise_on_error(resp)
        data = resp['Data']
        visibility = VisibilityMap.get(data['Visibility'])

        datahub_sts_url = f'{datahub_url}/ststoken?Revision={revision}'
        r_sts = self.session.get(url=datahub_sts_url, cookies=cookies,
                                 headers=self.builder_headers(self.headers))
        resp_sts = r_sts.json()
        raise_on_error(resp_sts)
        data_sts = resp_sts['Data']
        file_dir = visibility + '-unzipped' + '/' + namespace + '_' + dataset_name + '_' + zip_file_name
        data_sts['Dir'] = file_dir
        return data_sts

    def list_oss_dataset_objects(self, dataset_name, namespace, max_limit,
                                 is_recursive, is_filter_dir, revision, endpoint: Optional[str] = None):
        if not endpoint:
            endpoint = self.endpoint
        url = f'{endpoint}/api/v1/datasets/{namespace}/{dataset_name}/oss/tree/?' \
              f'MaxLimit={max_limit}&Revision={revision}&Recursive={is_recursive}&FilterDir={is_filter_dir}'

        cookies = ModelScopeConfig.get_cookies()
        resp = self.session.get(url=url, cookies=cookies, timeout=1800)
        resp = resp.json()
        raise_on_error(resp)
        resp = resp['Data']
        return resp

    def delete_oss_dataset_object(self, object_name: str, dataset_name: str,
                                  namespace: str, revision: str, endpoint: Optional[str] = None) -> str:
        if not object_name or not dataset_name or not namespace or not revision:
            raise ValueError('Args cannot be empty!')
        if not endpoint:
            endpoint = self.endpoint
        url = f'{endpoint}/api/v1/datasets/{namespace}/{dataset_name}/oss?Path={object_name}&Revision={revision}'

        cookies = ModelScopeConfig.get_cookies()
        resp = self.session.delete(url=url, cookies=cookies)
        resp = resp.json()
        raise_on_error(resp)
        resp = resp['Message']
        return resp

    def delete_oss_dataset_dir(self, object_name: str, dataset_name: str,
                               namespace: str, revision: str, endpoint: Optional[str] = None) -> str:
        if not object_name or not dataset_name or not namespace or not revision:
            raise ValueError('Args cannot be empty!')
        if not endpoint:
            endpoint = self.endpoint
        url = f'{endpoint}/api/v1/datasets/{namespace}/{dataset_name}/oss/prefix?Prefix={object_name}/' \
              f'&Revision={revision}'

        cookies = ModelScopeConfig.get_cookies()
        resp = self.session.delete(url=url, cookies=cookies)
        resp = resp.json()
        raise_on_error(resp)
        resp = resp['Message']
        return resp

    def datahub_remote_call(self, url):
        cookies = ModelScopeConfig.get_cookies()
        r = self.session.get(
            url,
            cookies=cookies,
            headers={'user-agent': ModelScopeConfig.get_user_agent()})
        resp = r.json()
        datahub_raise_on_error(url, resp, r)
        return resp['Data']

    def dataset_download_statistics(self, dataset_name: str, namespace: str,
                                    use_streaming: bool = False, endpoint: Optional[str] = None) -> None:
        is_ci_test = os.getenv('CI_TEST') == 'True'
        if not endpoint:
            endpoint = self.endpoint
        if dataset_name and namespace and not is_ci_test and not use_streaming:
            try:
                cookies = ModelScopeConfig.get_cookies()

                # Download count
                download_count_url = f'{endpoint}/api/v1/datasets/{namespace}/{dataset_name}/download/increase'
                download_count_resp = self.session.post(download_count_url, cookies=cookies,
                                                        headers=self.builder_headers(self.headers))
                raise_for_http_status(download_count_resp)

                # Download uv
                channel = DownloadChannel.LOCAL.value
                user_name = ''
                if MODELSCOPE_CLOUD_ENVIRONMENT in os.environ:
                    channel = os.environ[MODELSCOPE_CLOUD_ENVIRONMENT]
                if MODELSCOPE_CLOUD_USERNAME in os.environ:
                    user_name = os.environ[MODELSCOPE_CLOUD_USERNAME]
                download_uv_url = f'{endpoint}/api/v1/datasets/{namespace}/{dataset_name}/download/uv/' \
                                  f'{channel}?user={user_name}'

                download_uv_resp = self.session.post(download_uv_url, cookies=cookies,
                                                     headers=self.builder_headers(self.headers))
                download_uv_resp = download_uv_resp.json()
                raise_on_error(download_uv_resp)

            except Exception as e:
                logger.error(e)

    def builder_headers(self, headers):
        return {MODELSCOPE_REQUEST_ID: str(uuid.uuid4().hex),
                **headers}

    def get_file_base_path(self, repo_id: str, endpoint: Optional[str] = None) -> str:
        _namespace, _dataset_name = repo_id.split('/')
        if not endpoint:
            endpoint = self.endpoint
        return f'{endpoint}/api/v1/datasets/{_namespace}/{_dataset_name}/repo?'
        # return f'{endpoint}/api/v1/datasets/{namespace}/{dataset_name}/repo?Revision={revision}&FilePath='

    def create_repo(
            self,
            repo_id: str,
            *,
            token: Union[str, bool, None] = None,
            visibility: Optional[str] = Visibility.PUBLIC,
            repo_type: Optional[str] = REPO_TYPE_MODEL,
            chinese_name: Optional[str] = None,
            license: Optional[str] = Licenses.APACHE_V2,
            endpoint: Optional[str] = None,
            exist_ok: Optional[bool] = False,
            create_default_config: Optional[bool] = True,
            aigc_model: Optional[AigcModel] = None,
            **kwargs,
    ) -> str:
        """
        Create a repository on the ModelScope Hub.

        Args:
            repo_id (str): The repo id in the format of `owner_name/repo_name`.
            token (Union[str, bool, None]): The access token.
            visibility (Optional[str]): The visibility of the repo,
                could be `public`, `private`, `internal`, default to `public`.
            repo_type (Optional[str]): The repo type, default to `model`.
            chinese_name (Optional[str]): The Chinese name of the repo.
            license (Optional[str]): The license of the repo, default to `apache-2.0`.
            endpoint (Optional[str]): The endpoint to use.
                In the format of `https://www.modelscope.cn` or 'https://www.modelscope.ai'
            exist_ok (Optional[bool]): If the repo exists, whether to return the repo url directly.
            create_default_config (Optional[bool]): If True, create a default configuration file in the model repo.
            **kwargs: The additional arguments.

        Returns:
            str: The repo url.
        """

        if not repo_id:
            raise ValueError('Repo id cannot be empty!')
        if not endpoint:
            endpoint = self.endpoint

        self.login(access_token=token, endpoint=endpoint)

        repo_exists: bool = self.repo_exists(repo_id, repo_type=repo_type, endpoint=endpoint, token=token)
        if repo_exists:
            if exist_ok:
                repo_url: str = f'{endpoint}/{repo_type}s/{repo_id}'
                logger.warning(f'Repo {repo_id} already exists, got repo url: {repo_url}')
                return repo_url
            else:
                raise ValueError(f'Repo {repo_id} already exists!')

        repo_id_list = repo_id.split('/')
        if len(repo_id_list) != 2:
            raise ValueError('Invalid repo id, should be in the format of `owner_name/repo_name`')
        namespace, repo_name = repo_id_list

        if repo_type == REPO_TYPE_MODEL:
            visibilities = {k: v for k, v in ModelVisibility.__dict__.items() if not k.startswith('__')}
            visibility: int = visibilities.get(visibility.upper())
            if visibility is None:
                raise ValueError(f'Invalid visibility: {visibility}, '
                                 f'supported visibilities: `public`, `private`, `internal`')
            repo_url: str = self.create_model(
                model_id=repo_id,
                visibility=visibility,
                license=license,
                chinese_name=chinese_name,
                aigc_model=aigc_model
            )
            if create_default_config:
                with tempfile.TemporaryDirectory() as temp_cache_dir:
                    from modelscope.hub.repository import Repository
                    repo = Repository(temp_cache_dir, repo_id)
                    default_config = {
                        'framework': 'pytorch',
                        'task': 'text-generation',
                        'allow_remote': True
                    }
                    config_json = kwargs.get('config_json')
                    if not config_json:
                        config_json = {}
                    config = {**default_config, **config_json}
                    add_content_to_file(
                        repo,
                        'configuration.json', [json.dumps(config)],
                        ignore_push_error=True)
            print(f'New model created successfully at {repo_url}.', flush=True)

        elif repo_type == REPO_TYPE_DATASET:
            visibilities = {k: v for k, v in DatasetVisibility.__dict__.items() if not k.startswith('__')}
            visibility: int = visibilities.get(visibility.upper())
            if visibility is None:
                raise ValueError(f'Invalid visibility: {visibility}, '
                                 f'supported visibilities: `public`, `private`, `internal`')
            repo_url: str = self.create_dataset(
                dataset_name=repo_name,
                namespace=namespace,
                chinese_name=chinese_name,
                license=license,
                visibility=visibility,
            )
            print(f'New dataset created successfully at {repo_url}.', flush=True)

        else:
            raise ValueError(f'Invalid repo type: {repo_type}, supported repos: {REPO_TYPE_SUPPORT}')

        return repo_url

    def create_commit(
            self,
            repo_id: str,
            operations: Iterable[CommitOperation],
            *,
            commit_message: str,
            commit_description: Optional[str] = None,
            token: str = None,
            repo_type: Optional[str] = REPO_TYPE_MODEL,
            revision: Optional[str] = DEFAULT_REPOSITORY_REVISION,
            endpoint: Optional[str] = None,
            max_retries: int = 3,
            timeout: int = 180,
    ) -> CommitInfo:
        """
        Create a commit on the ModelScope Hub with retry mechanism.

        Args:
            repo_id (str): The repo id in the format of `owner_name/repo_name`.
            operations (Iterable[CommitOperation]): The commit operations.
            commit_message (str): The commit message.
            commit_description (Optional[str]): The commit description.
            token (str): The access token. If None, will use the cookies from the local cache.
                See `https://modelscope.cn/my/myaccesstoken` to get your token.
            repo_type (Optional[str]): The repo type, should be `model` or `dataset`. Defaults to `model`.
            revision (Optional[str]): The branch or tag name. Defaults to `DEFAULT_REPOSITORY_REVISION`.
            endpoint (Optional[str]): The endpoint to use.
                In the format of `https://www.modelscope.cn` or 'https://www.modelscope.ai'
            max_retries (int): Number of max retry attempts (default: 3).
            timeout (int): Timeout for each request in seconds (default: 180).

        Returns:
            CommitInfo: The commit info.

        Raises:
            requests.exceptions.RequestException: If all retry attempts fail.
        """
        if not repo_id:
            raise ValueError('Repo id cannot be empty!')

        if not endpoint:
            endpoint = self.endpoint

        if repo_type not in REPO_TYPE_SUPPORT:
            raise ValueError(f'Invalid repo type: {repo_type}, supported repos: {REPO_TYPE_SUPPORT}')

        url = f'{endpoint}/api/v1/repos/{repo_type}s/{repo_id}/commit/{revision}'
        commit_message = commit_message or f'Commit to {repo_id}'
        commit_description = commit_description or ''

        cookies = self.get_cookies(access_token=token, cookies_required=True)

        # Construct payload
        payload = self._prepare_commit_payload(
            operations=operations,
            commit_message=commit_message,
        )

        # POST with retry mechanism
        last_exception = None
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f'Attempt {attempt + 1} to create commit for {repo_id}...')
                response = requests.post(
                    url,
                    headers=self.builder_headers(self.headers),
                    data=json.dumps(payload),
                    cookies=cookies,
                    timeout=timeout,
                )

                if response.status_code != 200:
                    try:
                        error_detail = response.json()
                    except json.JSONDecodeError:
                        error_detail = response.text

                    error_msg = (
                        f'HTTP {response.status_code} error from {url}: '
                        f'{error_detail}'
                    )

                    # If server error (5xx), we can retry, otherwise (4xx) raise immediately
                    if 500 <= response.status_code < 600:
                        logger.warning(
                            f'Server error on attempt {attempt + 1}: {error_msg}'
                        )
                    else:
                        raise ValueError(f'Client request failed: {error_msg}')
                else:
                    resp = response.json()

                    oid = resp.get('Data', {}).get('oid', '')
                    logger.info(f'Commit succeeded: {url}')
                    return CommitInfo(
                        commit_url=url,
                        commit_message=commit_message,
                        commit_description=commit_description,
                        oid=oid,
                    )

            except requests.exceptions.RequestException as e:
                last_exception = e
                logger.warning(f'Request failed on attempt {attempt + 1}: {str(e)}')

            except Exception as e:
                last_exception = e
                logger.error(f'Unexpected error on attempt {attempt + 1}: {str(e)}')
                if attempt == max_retries - 1:
                    raise

            if attempt < max_retries - 1:
                time.sleep(1)

        # All retries exhausted
        raise requests.exceptions.RequestException(
            f'Failed to create commit after {max_retries} attempts. Last error: {last_exception}'
        )

    def upload_file(
            self,
            *,
            path_or_fileobj: Union[str, Path, bytes, BinaryIO],
            path_in_repo: str,
            repo_id: str,
            token: Union[str, None] = None,
            repo_type: Optional[str] = REPO_TYPE_MODEL,
            commit_message: Optional[str] = None,
            commit_description: Optional[str] = None,
            buffer_size_mb: Optional[int] = 1,
            tqdm_desc: Optional[str] = '[Uploading]',
            disable_tqdm: Optional[bool] = False,
            revision: Optional[str] = DEFAULT_REPOSITORY_REVISION
    ) -> CommitInfo:
        """
        Upload a file to the ModelScope Hub.

        Args:
            path_or_fileobj (Union[str, Path, bytes, BinaryIO]):
                The local file path or file-like object (BinaryIO) or bytes to upload.
            path_in_repo (str): The path in the repo to upload to.
            repo_id (str): The repo id in the format of `owner_name/repo_name`.
            token (Union[str, None]): The access token. If None, will use the cookies from the local cache.
                See `https://modelscope.cn/my/myaccesstoken` to get your token.
            repo_type (Optional[str]): The repo type, default to `model`.
            commit_message (Optional[str]): The commit message.
            commit_description (Optional[str]): The commit description.
            buffer_size_mb (Optional[int]): The buffer size in MB for reading the file. Default to 1MB.
            tqdm_desc (Optional[str]): The description for the tqdm progress bar. Default to '[Uploading]'.
            disable_tqdm (Optional[bool]): Whether to disable the tqdm progress bar. Default to False.
            revision (Optional[str]): The branch or tag name. Defaults to `DEFAULT_REPOSITORY_REVISION`.

        Returns:
            CommitInfo: The commit info.

        Examples:
            >>> from modelscope.hub.api import HubApi
            >>> api = HubApi()
            >>> commit_info = api.upload_file(
            ...     path_or_fileobj='/path/to/your/file.txt',
            ...     path_in_repo='optional/path/in/repo/file.txt',
            ...     repo_id='your-namespace/your-repo-name',
            ...     commit_message='Upload file.txt to ModelScope hub'
            ... )
            >>> print(commit_info)
        """

        if repo_type not in REPO_TYPE_SUPPORT:
            raise ValueError(f'Invalid repo type: {repo_type}, supported repos: {REPO_TYPE_SUPPORT}')

        if not path_or_fileobj:
            raise ValueError('Path or file object cannot be empty!')

        # Check authentication first
        self.get_cookies(access_token=token, cookies_required=True)

        if isinstance(path_or_fileobj, (str, Path)):
            path_or_fileobj = os.path.abspath(os.path.expanduser(path_or_fileobj))
            path_in_repo = path_in_repo or os.path.basename(path_or_fileobj)
        else:
            # If path_or_fileobj is bytes or BinaryIO, then path_in_repo must be provided
            if not path_in_repo:
                raise ValueError('Arg `path_in_repo` cannot be empty!')

        # Read file content if path_or_fileobj is a file-like object (BinaryIO)
        # TODO: to be refined
        if isinstance(path_or_fileobj, io.BufferedIOBase):
            path_or_fileobj = path_or_fileobj.read()

        self.upload_checker.check_file(path_or_fileobj)
        self.upload_checker.check_normal_files(
            file_path_list=[path_or_fileobj],
            repo_type=repo_type,
        )

        commit_message = (
            commit_message if commit_message is not None else f'Upload {path_in_repo} to ModelScope hub'
        )

        if buffer_size_mb <= 0:
            raise ValueError('Buffer size: `buffer_size_mb` must be greater than 0')

        hash_info_d: dict = get_file_hash(
            file_path_or_obj=path_or_fileobj,
            buffer_size_mb=buffer_size_mb,
        )
        file_size: int = hash_info_d['file_size']
        file_hash: str = hash_info_d['file_hash']

        self.create_repo(repo_id=repo_id,
                         token=token,
                         repo_type=repo_type,
                         endpoint=self.endpoint,
                         exist_ok=True,
                         create_default_config=False)

        upload_res: dict = self._upload_blob(
            repo_id=repo_id,
            repo_type=repo_type,
            sha256=file_hash,
            size=file_size,
            data=path_or_fileobj,
            disable_tqdm=disable_tqdm,
            tqdm_desc=tqdm_desc,
        )

        # Construct commit info and create commit
        add_operation: CommitOperationAdd = CommitOperationAdd(
            path_in_repo=path_in_repo,
            path_or_fileobj=path_or_fileobj,
            file_hash_info=hash_info_d,
        )
        add_operation._upload_mode = 'lfs' if self.upload_checker.is_lfs(path_or_fileobj, repo_type) else 'normal'
        add_operation._is_uploaded = upload_res['is_uploaded']
        operations = [add_operation]

        print(f'Committing file to {repo_id} ...', flush=True)
        commit_info: CommitInfo = self.create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message=commit_message,
            commit_description=commit_description,
            token=token,
            repo_type=repo_type,
            revision=revision,
        )

        return commit_info

    def upload_folder(
            self,
            *,
            repo_id: str,
            folder_path: Union[str, Path, List[str], List[Path]],
            path_in_repo: Optional[str] = '',
            commit_message: Optional[str] = None,
            commit_description: Optional[str] = None,
            token: Union[str, None] = None,
            repo_type: Optional[str] = REPO_TYPE_MODEL,
            allow_patterns: Optional[Union[List[str], str]] = None,
            ignore_patterns: Optional[Union[List[str], str]] = None,
            max_workers: int = DEFAULT_MAX_WORKERS,
            revision: Optional[str] = DEFAULT_REPOSITORY_REVISION,
    ) -> Union[CommitInfo, List[CommitInfo]]:
        """
        Upload a folder to the ModelScope Hub.

        Args:
            repo_id (str): The repo id in the format of `owner_name/repo_name`.
            folder_path (Union[str, Path, List[str], List[Path]]): The folder path or list of file paths to upload.
            path_in_repo (Optional[str]): The path in the repo to upload to.
            commit_message (Optional[str]): The commit message.
            commit_description (Optional[str]): The commit description.
            token (Union[str, None]): The access token. If None, will use the cookies from the local cache.
                See `https://modelscope.cn/my/myaccesstoken` to get your token.
            repo_type (Optional[str]): The repo type, default to `model`.
            allow_patterns (Optional[Union[List[str], str]]): The patterns to allow.
            ignore_patterns (Optional[Union[List[str], str]]): The patterns to ignore.
            max_workers (int): The maximum number of workers to use for uploading files concurrently.
                Defaults to `DEFAULT_MAX_WORKERS`.
            revision (Optional[str]): The branch or tag name. Defaults to `DEFAULT_REPOSITORY_REVISION`.

        Returns:
            Union[CommitInfo, List[CommitInfo]]:
                The commit info or list of commit infos if multiple batches are committed.

        Examples:
            >>> from modelscope.hub.api import HubApi
            >>> api = HubApi()
            >>> commit_info = api.upload_folder(
            ...     repo_id='your-namespace/your-repo-name',
            ...     folder_path='/path/to/your/folder',
            ...     path_in_repo='optional/path/in/repo',
            ...     commit_message='Upload my folder',
            ...     token='your-access-token'
            ... )
            >>> print(commit_info.commit_url)
        """
        if not repo_id:
            raise ValueError('The arg `repo_id` cannot be empty!')

        if folder_path is None:
            raise ValueError('The arg `folder_path` cannot be None!')

        if repo_type not in REPO_TYPE_SUPPORT:
            raise ValueError(f'Invalid repo type: {repo_type}, supported repos: {REPO_TYPE_SUPPORT}')

        # Check authentication first
        self.get_cookies(access_token=token, cookies_required=True)

        allow_patterns = allow_patterns if allow_patterns else None
        ignore_patterns = ignore_patterns if ignore_patterns else None

        # Ignore .git folder
        if ignore_patterns is None:
            ignore_patterns = []
        elif isinstance(ignore_patterns, str):
            ignore_patterns = [ignore_patterns]
        ignore_patterns += DEFAULT_IGNORE_PATTERNS

        commit_message = (
            commit_message if commit_message is not None else f'Upload to {repo_id} on ModelScope hub'
        )
        commit_description = commit_description or 'Uploading files'

        # Get the list of files to upload, e.g. [('data/abc.png', '/path/to/abc.png'), ...]
        logger.info('Preparing files to upload ...')
        prepared_repo_objects = self._prepare_upload_folder(
            folder_path_or_files=folder_path,
            path_in_repo=path_in_repo,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
        if len(prepared_repo_objects) == 0:
            raise ValueError(f'No files to upload in the folder: {folder_path} !')

        logger.info(f'Checking {len(prepared_repo_objects)} files to upload ...')
        self.upload_checker.check_normal_files(
            file_path_list=[item for _, item in prepared_repo_objects],
            repo_type=repo_type,
        )

        self.create_repo(repo_id=repo_id,
                         token=token,
                         repo_type=repo_type,
                         endpoint=self.endpoint,
                         exist_ok=True,
                         create_default_config=False)

        @thread_executor(max_workers=max_workers, disable_tqdm=False)
        def _upload_items(item_pair, **kwargs):
            file_path_in_repo, file_path = item_pair

            hash_info_d: dict = get_file_hash(
                file_path_or_obj=file_path,
            )
            file_size: int = hash_info_d['file_size']
            file_hash: str = hash_info_d['file_hash']

            upload_res: dict = self._upload_blob(
                repo_id=repo_id,
                repo_type=repo_type,
                sha256=file_hash,
                size=file_size,
                data=file_path,
                disable_tqdm=file_size <= UPLOAD_BLOB_TQDM_DISABLE_THRESHOLD,
                tqdm_desc='[Uploading ' + file_path_in_repo + ']',
            )

            return {
                'file_path_in_repo': file_path_in_repo,
                'file_path': file_path,
                'is_uploaded': upload_res['is_uploaded'],
                'file_hash_info': hash_info_d,
            }

        uploaded_items_list = _upload_items(
            prepared_repo_objects,
            repo_id=repo_id,
            token=token,
            repo_type=repo_type,
            commit_message=commit_message,
            commit_description=commit_description,
            buffer_size_mb=1,
            disable_tqdm=False,
        )

        # Construct commit info and create commit
        operations = []

        for item_d in uploaded_items_list:
            prepared_path_in_repo: str = item_d['file_path_in_repo']
            prepared_file_path: str = item_d['file_path']
            is_uploaded: bool = item_d['is_uploaded']
            file_hash_info: dict = item_d['file_hash_info']
            opt = CommitOperationAdd(
                path_in_repo=prepared_path_in_repo,
                path_or_fileobj=prepared_file_path,
                file_hash_info=file_hash_info,
            )

            # check normal or lfs
            opt._upload_mode = 'lfs' if self.upload_checker.is_lfs(prepared_file_path, repo_type) else 'normal'
            opt._is_uploaded = is_uploaded
            operations.append(opt)

        if len(operations) == 0:
            raise ValueError(f'No files to upload in the folder: {folder_path} !')

        # Commit the operations in batches
        commit_batch_size: int = UPLOAD_COMMIT_BATCH_SIZE if UPLOAD_COMMIT_BATCH_SIZE > 0 else len(operations)
        num_batches = (len(operations) - 1) // commit_batch_size + 1
        print(f'Committing {len(operations)} files in {num_batches} batch(es) of size {commit_batch_size}.',
              flush=True)
        commit_infos: List[CommitInfo] = []
        for i in tqdm(range(num_batches), desc='[Committing batches] ', total=num_batches):
            batch_operations = operations[i * commit_batch_size: (i + 1) * commit_batch_size]
            batch_commit_message = f'{commit_message} (batch {i + 1}/{num_batches})'

            commit_info: CommitInfo = self.create_commit(
                repo_id=repo_id,
                operations=batch_operations,
                commit_message=batch_commit_message,
                commit_description=commit_description,
                token=token,
                repo_type=repo_type,
                revision=revision,
            )
            commit_infos.append(commit_info)

        return commit_infos[0] if len(commit_infos) == 1 else commit_infos

    def _upload_blob(
            self,
            *,
            repo_id: str,
            repo_type: str,
            sha256: str,
            size: int,
            data: Union[str, Path, bytes, BinaryIO],
            disable_tqdm: Optional[bool] = False,
            tqdm_desc: Optional[str] = '[Uploading]',
            buffer_size_mb: Optional[int] = 1,
    ) -> dict:

        res_d: dict = dict(
            url=None,
            is_uploaded=False,
            status_code=None,
            status_msg=None,
        )

        objects = [{'oid': sha256, 'size': size}]
        upload_objects = self._validate_blob(
            repo_id=repo_id,
            repo_type=repo_type,
            objects=objects,
        )

        # upload_object: {'url': 'xxx', 'oid': 'xxx'}
        upload_object = upload_objects[0] if len(upload_objects) == 1 else None

        if upload_object is None:
            logger.debug(f'Blob {sha256[:8]} has already uploaded, reuse it.')
            res_d['is_uploaded'] = True
            return res_d

        cookies = ModelScopeConfig.get_cookies()
        cookies = dict(cookies) if cookies else None
        if cookies is None:
            raise ValueError('Token does not exist, please login first.')

        self.headers.update({'Cookie': f"m_session_id={cookies['m_session_id']}"})
        headers = self.builder_headers(self.headers)

        def read_in_chunks(file_object, pbar, chunk_size=buffer_size_mb * 1024 * 1024):
            """Lazy function (generator) to read a file piece by piece."""
            while True:
                ck = file_object.read(chunk_size)
                if not ck:
                    break
                pbar.update(len(ck))
                yield ck

        with tqdm(
                total=size,
                unit='B',
                unit_scale=True,
                desc=tqdm_desc,
                disable=disable_tqdm
        ) as pbar:

            if isinstance(data, (str, Path)):
                with open(data, 'rb') as f:
                    response = requests.put(
                        upload_object['url'],
                        headers=headers,
                        data=read_in_chunks(f, pbar)
                    )

            elif isinstance(data, bytes):
                response = requests.put(
                    upload_object['url'],
                    headers=headers,
                    data=read_in_chunks(io.BytesIO(data), pbar)
                )

            elif isinstance(data, io.BufferedIOBase):
                response = requests.put(
                    upload_object['url'],
                    headers=headers,
                    data=read_in_chunks(data, pbar)
                )

            else:
                raise ValueError('Invalid data type to upload')

        raise_for_http_status(rsp=response)
        resp = response.json()
        raise_on_error(rsp=resp)

        res_d['url'] = upload_object['url']
        res_d['status_code'] = resp['Code']
        res_d['status_msg'] = resp['Message']

        return res_d

    def _validate_blob(
            self,
            *,
            repo_id: str,
            repo_type: str,
            objects: List[Dict[str, Any]],
            endpoint: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Check the blob has already uploaded.
        True -- uploaded; False -- not uploaded.

        Args:
            repo_id (str): The repo id ModelScope.
            repo_type (str): The repo type. `dataset`, `model`, etc.
            objects (List[Dict[str, Any]]): The objects to check.
                oid (str): The sha256 hash value.
                size (int): The size of the blob.
            endpoint: the endpoint to use, default to None to use endpoint specified in the class

        Returns:
            List[Dict[str, Any]]: The result of the check.
        """

        # construct URL
        if not endpoint:
            endpoint = self.endpoint
        url = f'{endpoint}/api/v1/repos/{repo_type}s/{repo_id}/info/lfs/objects/batch'

        # build payload
        payload = {
            'operation': 'upload',
            'objects': objects,
        }

        cookies = ModelScopeConfig.get_cookies()
        if cookies is None:
            raise ValueError('Token does not exist, please login first.')
        response = requests.post(
            url,
            headers=self.builder_headers(self.headers),
            data=json.dumps(payload),
            cookies=cookies
        )

        raise_for_http_status(rsp=response)
        resp = response.json()
        raise_on_error(rsp=resp)

        upload_objects = []  # list of objects to upload, [{'url': 'xxx', 'oid': 'xxx'}, ...]
        resp_objects = resp['Data']['objects']
        for obj in resp_objects:
            upload_objects.append(
                {'url': obj['actions']['upload']['href'],
                 'oid': obj['oid']}
            )

        return upload_objects

    def _prepare_upload_folder(
            self,
            folder_path_or_files: Union[str, Path, List[str], List[Path]],
            path_in_repo: str,
            allow_patterns: Optional[Union[List[str], str]] = None,
            ignore_patterns: Optional[Union[List[str], str]] = None,
    ) -> List[Union[tuple, list]]:
        folder_path = None
        files_path = None
        if isinstance(folder_path_or_files, list):
            if os.path.isfile(folder_path_or_files[0]):
                files_path = folder_path_or_files
            else:
                raise ValueError('Uploading multiple folders is not supported now.')
        else:
            if os.path.isfile(folder_path_or_files):
                files_path = [folder_path_or_files]
            else:
                folder_path = folder_path_or_files

        if files_path is None:
            self.upload_checker.check_folder(folder_path)
            folder_path = Path(folder_path).expanduser().resolve()
            if not folder_path.is_dir():
                raise ValueError(f"Provided path: '{folder_path}' is not a directory")

            # List files from folder
            relpath_to_abspath = {
                path.relative_to(folder_path).as_posix(): path
                for path in sorted(folder_path.glob('**/*'))  # sorted to be deterministic
                if path.is_file()
            }
        else:
            relpath_to_abspath = {}
            for path in files_path:
                if os.path.isfile(path):
                    self.upload_checker.check_file(path)
                    relpath_to_abspath[os.path.basename(path)] = path

        # Filter files
        filtered_repo_objects = list(
            RepoUtils.filter_repo_objects(
                relpath_to_abspath.keys(), allow_patterns=allow_patterns, ignore_patterns=ignore_patterns
            )
        )

        prefix = f"{path_in_repo.strip('/')}/" if path_in_repo else ''

        prepared_repo_objects = [
            (prefix + relpath, str(relpath_to_abspath[relpath]))
            for relpath in filtered_repo_objects
        ]

        logger.info(f'Prepared {len(prepared_repo_objects)} files for upload.')

        return prepared_repo_objects

    @staticmethod
    def _prepare_commit_payload(
            operations: Iterable[CommitOperation],
            commit_message: str,
    ) -> Dict[str, Any]:
        """
        Prepare the commit payload to be sent to the ModelScope hub.
        """

        payload = {
            'commit_message': commit_message,
            'actions': []
        }

        nb_ignored_files = 0

        # 2. Send operations, one per line
        for operation in operations:

            # Skip ignored files
            if isinstance(operation, CommitOperationAdd) and operation._should_ignore:
                logger.debug(f"Skipping file '{operation.path_in_repo}' in commit (ignored by gitignore file).")
                nb_ignored_files += 1
                continue

            # 2.a. Case adding a normal file
            if isinstance(operation, CommitOperationAdd) and operation._upload_mode == 'normal':

                commit_action = {
                    'action': 'update' if operation._is_uploaded else 'create',
                    'path': operation.path_in_repo,
                    'type': 'normal',
                    'size': operation.upload_info.size,
                    'sha256': '',
                    'content': operation.b64content().decode(),
                    'encoding': 'base64',
                }
                payload['actions'].append(commit_action)

            # 2.b. Case adding an LFS file
            elif isinstance(operation, CommitOperationAdd) and operation._upload_mode == 'lfs':

                commit_action = {
                    'action': 'update' if operation._is_uploaded else 'create',
                    'path': operation.path_in_repo,
                    'type': 'lfs',
                    'size': operation.upload_info.size,
                    'sha256': operation.upload_info.sha256,
                    'content': '',
                    'encoding': '',
                }
                payload['actions'].append(commit_action)

            else:
                raise ValueError(
                    f'Unknown operation to commit. Operation: {operation}. Upload mode:'
                    f" {getattr(operation, '_upload_mode', None)}"
                )

        if nb_ignored_files > 0:
            logger.info(f'Skipped {nb_ignored_files} file(s) in commit (ignored by gitignore file).')

        return payload

    def _get_internal_acceleration_domain(self, internal_timeout: float = 0.2):
        """
        Get the internal acceleration domain.

        Args:
            internal_timeout (float): The timeout for the request. Default to 0.2s

        Returns:
            str: The internal acceleration domain. e.g. `cn-hangzhou`, `cn-zhangjiakou`
        """

        def send_request(url: str, timeout: float):
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
            except requests.exceptions.RequestException:
                response = None

            return response

        internal_url = f'{self.endpoint}/api/v1/repos/internalAccelerationInfo'

        # Get internal url and region for acceleration
        internal_info_response = send_request(url=internal_url, timeout=internal_timeout)
        region_id: str = ''
        if internal_info_response is not None:
            internal_info_response = internal_info_response.json()
            if 'Data' in internal_info_response:
                query_addr = internal_info_response['Data']['InternalRegionQueryAddress']
            else:
                query_addr: str = ''

            if query_addr:
                domain_response = send_request(query_addr, timeout=internal_timeout)
                if domain_response is not None:
                    region_id = domain_response.text.strip()

        return region_id

    def delete_files(self,
                     repo_id: str,
                     repo_type: str,
                     delete_patterns: Union[str, List[str]],
                     *,
                     revision: Optional[str] = DEFAULT_MODEL_REVISION,
                     endpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Delete files in batch using glob (wildcard) patterns, e.g. '*.py', 'data/*.csv', 'foo*', etc.

        Example:
            # Delete all Python and Markdown files in a model repo
            api.delete_files(
                repo_id='your_username/your_model',
                repo_type=REPO_TYPE_MODEL,
                delete_patterns=['*.py', '*.md']
            )

            # Delete all CSV files in the data/ directory of a dataset repo
            api.delete_files(
                repo_id='your_username/your_dataset',
                repo_type=REPO_TYPE_DATASET,
                delete_patterns='data/*.csv'
            )

        Args:
            repo_id (str): 'owner/repo_name' or 'owner/dataset_name', e.g. 'Koko/my_model'
            repo_type (str): REPO_TYPE_MODEL or REPO_TYPE_DATASET
            delete_patterns (str or List[str]): List of glob patterns, e.g. '*.py', 'data/*.csv', 'foo*'
            revision (str, optional): Branch or tag name
            endpoint (str, optional): API endpoint
        Returns:
            dict: Deletion result
        """
        if repo_type not in REPO_TYPE_SUPPORT:
            raise ValueError(f'Unsupported repo_type: {repo_type}')
        if not delete_patterns:
            raise ValueError('delete_patterns cannot be empty')
        if isinstance(delete_patterns, str):
            delete_patterns = [delete_patterns]

        cookies = ModelScopeConfig.get_cookies()
        if not endpoint:
            endpoint = self.endpoint
        if cookies is None:
            raise ValueError('Token does not exist, please login first.')
        headers = self.builder_headers(self.headers)

        # List all files in the repo
        if repo_type == REPO_TYPE_MODEL:
            files = self.get_model_files(
                repo_id,
                revision=revision or DEFAULT_MODEL_REVISION,
                recursive=True,
                endpoint=endpoint,
                use_cookies=cookies,
            )
            file_paths = [f['Path'] for f in files]
        elif repo_type == REPO_TYPE_DATASET:
            file_paths = []
            page_number = 1
            page_size = 100
            while True:
                try:
                    dataset_files: List[Dict[str, Any]] = self.get_dataset_files(
                        repo_id=repo_id,
                        revision=revision or DEFAULT_DATASET_REVISION,
                        recursive=True,
                        page_number=page_number,
                        page_size=page_size,
                        endpoint=endpoint,
                    )
                except Exception as e:
                    logger.error(f'Get dataset: {repo_id} file list failed, message: {str(e)}')
                    break

                # Parse data (Type: 'tree' or 'blob')
                for file_info_d in dataset_files:
                    if file_info_d['Type'] != 'tree':
                        file_paths.append(file_info_d['Path'])

                if len(dataset_files) < page_size:
                    break

                page_number += 1
        else:
            raise ValueError(f'Unsupported repo_type: {repo_type}, supported repos: {REPO_TYPE_SUPPORT}')

        # Glob pattern matching
        to_delete = []
        for path in file_paths:
            for delete_pattern in delete_patterns:
                if fnmatch.fnmatch(path, delete_pattern):
                    to_delete.append(path)
                    break

        deleted_files, failed_files = [], []
        for path in to_delete:
            try:
                if repo_type == REPO_TYPE_MODEL:
                    owner, repo_name = repo_id.split('/')
                    url = f'{endpoint}/api/v1/models/{owner}/{repo_name}/file'
                    params = {
                        'Revision': revision or DEFAULT_MODEL_REVISION,
                        'FilePath': path
                    }
                elif repo_type == REPO_TYPE_DATASET:
                    owner, dataset_name = repo_id.split('/')
                    url = f'{endpoint}/api/v1/datasets/{owner}/{dataset_name}/repo'
                    params = {
                        'FilePath': path
                    }
                else:
                    raise ValueError(f'Unsupported repo_type: {repo_type}, supported repos: {REPO_TYPE_SUPPORT}')

                r = self.session.delete(url, params=params, cookies=cookies, headers=headers)
                raise_for_http_status(r)
                resp = r.json()
                raise_on_error(resp)
                deleted_files.append(path)
            except Exception as e:
                failed_files.append(path)
                logger.error(f'Failed to delete {path}: {str(e)}')

        return {
            'deleted_files': deleted_files,
            'failed_files': failed_files,
            'total_files': len(to_delete)
        }


class ModelScopeConfig:
    path_credential = expanduser(MODELSCOPE_CREDENTIALS_PATH)
    COOKIES_FILE_NAME = 'cookies'
    GIT_TOKEN_FILE_NAME = 'git_token'
    USER_INFO_FILE_NAME = 'user'
    USER_SESSION_ID_FILE_NAME = 'session'
    cookie_expired_warning = False

    @staticmethod
    def make_sure_credential_path_exist():
        os.makedirs(ModelScopeConfig.path_credential, exist_ok=True)

    @staticmethod
    def save_cookies(cookies: CookieJar):
        ModelScopeConfig.make_sure_credential_path_exist()
        with open(
                os.path.join(ModelScopeConfig.path_credential,
                             ModelScopeConfig.COOKIES_FILE_NAME), 'wb+') as f:
            pickle.dump(cookies, f)

    @staticmethod
    def get_cookies():
        cookies_path = os.path.join(ModelScopeConfig.path_credential,
                                    ModelScopeConfig.COOKIES_FILE_NAME)
        if os.path.exists(cookies_path):
            with open(cookies_path, 'rb') as f:
                cookies = pickle.load(f)
                for cookie in cookies:
                    if cookie.name == 'm_session_id' and cookie.is_expired() and \
                            not ModelScopeConfig.cookie_expired_warning:
                        ModelScopeConfig.cookie_expired_warning = True
                        logger.info('Not logged-in, you can login for uploading'
                                    'or accessing controlled entities.')
                        return None
                return cookies
        return None

    @staticmethod
    def get_user_session_id():
        session_path = os.path.join(ModelScopeConfig.path_credential,
                                    ModelScopeConfig.USER_SESSION_ID_FILE_NAME)
        session_id = ''
        if os.path.exists(session_path):
            with open(session_path, 'rb') as f:
                session_id = str(f.readline().strip(), encoding='utf-8')
                return session_id
        if session_id == '' or len(session_id) != 32:
            session_id = str(uuid.uuid4().hex)
            ModelScopeConfig.make_sure_credential_path_exist()
            with open(session_path, 'w+') as wf:
                wf.write(session_id)

        return session_id

    @staticmethod
    def save_token(token: str):
        ModelScopeConfig.make_sure_credential_path_exist()
        with open(
                os.path.join(ModelScopeConfig.path_credential,
                             ModelScopeConfig.GIT_TOKEN_FILE_NAME), 'w+') as f:
            f.write(token)

    @staticmethod
    def save_user_info(user_name: str, user_email: str):
        ModelScopeConfig.make_sure_credential_path_exist()
        with open(
                os.path.join(ModelScopeConfig.path_credential,
                             ModelScopeConfig.USER_INFO_FILE_NAME), 'w+') as f:
            f.write('%s:%s' % (user_name, user_email))

    @staticmethod
    def get_user_info() -> Tuple[str, str]:
        try:
            with open(
                    os.path.join(ModelScopeConfig.path_credential,
                                 ModelScopeConfig.USER_INFO_FILE_NAME),
                    'r',
                    encoding='utf-8') as f:
                info = f.read()
                return info.split(':')[0], info.split(':')[1]
        except FileNotFoundError:
            pass
        return None, None

    @staticmethod
    def get_token() -> Optional[str]:
        """
        Get token or None if not existent.

        Returns:
            `str` or `None`: The token, `None` if it doesn't exist.

        """
        token = None
        try:
            with open(
                    os.path.join(ModelScopeConfig.path_credential,
                                 ModelScopeConfig.GIT_TOKEN_FILE_NAME),
                    'r',
                    encoding='utf-8') as f:
                token = f.read()
        except FileNotFoundError:
            pass
        return token

    @staticmethod
    def get_user_agent(user_agent: Union[Dict, str, None] = None, ) -> str:
        """Formats a user-agent string with basic info about a request.

        Args:
            user_agent (`str`, `dict`, *optional*):
                The user agent info in the form of a dictionary or a single string.

        Returns:
            The formatted user-agent string.
        """

        # include some more telemetrics when executing in dedicated
        # cloud containers
        env = 'custom'
        if MODELSCOPE_CLOUD_ENVIRONMENT in os.environ:
            env = os.environ[MODELSCOPE_CLOUD_ENVIRONMENT]
        user_name = 'unknown'
        if MODELSCOPE_CLOUD_USERNAME in os.environ:
            user_name = os.environ[MODELSCOPE_CLOUD_USERNAME]

        from modelscope import __version__
        ua = 'modelscope/%s; python/%s; session_id/%s; platform/%s; processor/%s; env/%s; user/%s' % (
            __version__,
            platform.python_version(),
            ModelScopeConfig.get_user_session_id(),
            platform.platform(),
            platform.processor(),
            env,
            user_name,
        )
        if isinstance(user_agent, dict):
            ua += '; ' + '; '.join(f'{k}/{v}' for k, v in user_agent.items())
        elif isinstance(user_agent, str):
            ua += '; ' + user_agent
        return ua


class UploadingCheck:
    """
    Check the files and folders to be uploaded.

    Args:
        max_file_count (int): The maximum number of files to be uploaded. Default to `UPLOAD_MAX_FILE_COUNT`.
        max_file_count_in_dir (int): The maximum number of files in a directory.
            Default to `UPLOAD_MAX_FILE_COUNT_IN_DIR`.
        max_file_size (int): The maximum size of a single file in bytes. Default to `UPLOAD_MAX_FILE_SIZE`.
        size_threshold_to_enforce_lfs (int): The size threshold to enforce LFS in bytes.
            Files larger than this size will be enforced to be uploaded via LFS.
            Default to `UPLOAD_SIZE_THRESHOLD_TO_ENFORCE_LFS`.
        normal_file_size_total_limit (int): The total size limit of normal files in bytes.
            Default to `UPLOAD_NORMAL_FILE_SIZE_TOTAL_LIMIT`.

    Examples:
        >>> from modelscope.hub.api import UploadingCheck
        >>> upload_checker = UploadingCheck()
        >>> upload_checker.check_file('/path/to/your/file.txt')
        >>> upload_checker.check_folder('/path/to/your/folder')
        >>> is_lfs = upload_checker.is_lfs('/path/to/your/file.txt', repo_type='model')
        >>> print(f'Is LFS: {is_lfs}')
    """
    def __init__(
            self,
            max_file_count: int = UPLOAD_MAX_FILE_COUNT,
            max_file_count_in_dir: int = UPLOAD_MAX_FILE_COUNT_IN_DIR,
            max_file_size: int = UPLOAD_MAX_FILE_SIZE,
            size_threshold_to_enforce_lfs: int = UPLOAD_SIZE_THRESHOLD_TO_ENFORCE_LFS,
            normal_file_size_total_limit: int = UPLOAD_NORMAL_FILE_SIZE_TOTAL_LIMIT,
    ):
        self.max_file_count = max_file_count
        self.max_file_count_in_dir = max_file_count_in_dir
        self.max_file_size = max_file_size
        self.size_threshold_to_enforce_lfs = size_threshold_to_enforce_lfs
        self.normal_file_size_total_limit = normal_file_size_total_limit

    def check_file(self, file_path_or_obj) -> None:
        """
        Check a single file to be uploaded.

        Args:
            file_path_or_obj (Union[str, Path, bytes, BinaryIO]): The file path or file-like object to be checked.

        Raises:
            ValueError: If the file does not exist or exceeds the size limit.
        """
        if isinstance(file_path_or_obj, (str, Path)):
            if not os.path.exists(file_path_or_obj):
                raise ValueError(f'File {file_path_or_obj} does not exist')

        file_size: int = get_file_size(file_path_or_obj)
        if file_size > self.max_file_size:
            logger.warning(f'File exceeds size limit: {self.max_file_size / (1024 ** 3)} GB, '
                           f'got {round(file_size / (1024 ** 3), 4)} GB')

    def check_folder(self, folder_path: Union[str, Path]):
        """
        Check a folder to be uploaded.

        Args:
            folder_path (Union[str, Path]): The folder path to be checked.

        Raises:
            ValueError: If the folder does not exist or exceeds the file count limit.
        """
        file_count = 0
        dir_count = 0

        if isinstance(folder_path, str):
            folder_path = Path(folder_path)

        for item in folder_path.iterdir():
            if item.is_file():
                file_count += 1
                item_size: int = get_file_size(item)
                if item_size > self.max_file_size:
                    logger.warning(f'File {item} exceeds size limit: {self.max_file_size / (1024 ** 3)} GB',
                                   f'got {round(item_size / (1024 ** 3), 4)} GB')
            elif item.is_dir():
                dir_count += 1
                # Count items in subdirectories recursively
                sub_file_count, sub_dir_count = self.check_folder(item)
                if (sub_file_count + sub_dir_count) > self.max_file_count_in_dir:
                    raise ValueError(f'Directory {item} contains {sub_file_count + sub_dir_count} items '
                                     f'and exceeds limit: {self.max_file_count_in_dir}')
                file_count += sub_file_count
                dir_count += sub_dir_count

        if file_count > self.max_file_count:
            raise ValueError(f'Total file count {file_count} and exceeds limit: {self.max_file_count}')

        return file_count, dir_count

    def is_lfs(self, file_path_or_obj: Union[str, Path, bytes, BinaryIO], repo_type: str) -> bool:
        """
        Check if a file should be uploaded via LFS.

        Args:
            file_path_or_obj (Union[str, Path, bytes, BinaryIO]): The file path or file-like object to be checked.
            repo_type (str): The repo type, either `model` or `dataset`.

        Returns:
            bool: True if the file should be uploaded via LFS, False otherwise.
        """
        hit_lfs_suffix = True

        if isinstance(file_path_or_obj, (str, Path)):
            file_path_or_obj = Path(file_path_or_obj)
            if not file_path_or_obj.exists():
                raise ValueError(f'File {file_path_or_obj} does not exist')

            if repo_type == REPO_TYPE_MODEL:
                if file_path_or_obj.suffix not in MODEL_LFS_SUFFIX:
                    hit_lfs_suffix = False
            elif repo_type == REPO_TYPE_DATASET:
                if file_path_or_obj.suffix not in DATASET_LFS_SUFFIX:
                    hit_lfs_suffix = False
            else:
                raise ValueError(f'Invalid repo type: {repo_type}, supported repos: {REPO_TYPE_SUPPORT}')

        file_size: int = get_file_size(file_path_or_obj)

        return file_size > self.size_threshold_to_enforce_lfs or hit_lfs_suffix

    def check_normal_files(self, file_path_list: List[Union[str, Path]], repo_type: str) -> None:
        """
        Check a list of normal files to be uploaded.

        Args:
            file_path_list (List[Union[str, Path]]): The list of file paths to be checked.
            repo_type (str): The repo type, either `model` or `dataset`.

        Raises:
            ValueError: If the total size of normal files exceeds the limit.

        Returns: None
        """
        normal_file_list = [item for item in file_path_list if not self.is_lfs(item, repo_type)]
        total_size = sum([get_file_size(item) for item in normal_file_list])

        if total_size > self.normal_file_size_total_limit:
            raise ValueError(f'Total size of non-lfs files {total_size / (1024 * 1024)}MB '
                             f'and exceeds limit: {self.normal_file_size_total_limit / (1024 * 1024)}MB')
