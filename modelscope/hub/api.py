# Copyright (c) Alibaba, Inc. and its affiliates.
# yapf: disable

import datetime
import functools
import os
import pickle
import platform
import shutil
import tempfile
import uuid
from collections import defaultdict
from http import HTTPStatus
from http.cookiejar import CookieJar
from os.path import expanduser
from typing import Dict, List, Optional, Tuple, Union

from requests import Session
from requests.adapters import HTTPAdapter, Retry

from modelscope import __version__
from modelscope.hub.constants import (API_HTTP_CLIENT_TIMEOUT,
                                      API_RESPONSE_FIELD_DATA,
                                      API_RESPONSE_FIELD_EMAIL,
                                      API_RESPONSE_FIELD_GIT_ACCESS_TOKEN,
                                      API_RESPONSE_FIELD_MESSAGE,
                                      API_RESPONSE_FIELD_USERNAME,
                                      DEFAULT_CREDENTIALS_PATH,
                                      MODELSCOPE_CLOUD_ENVIRONMENT,
                                      MODELSCOPE_CLOUD_USERNAME,
                                      ONE_YEAR_SECONDS,
                                      REQUESTS_API_HTTP_METHOD, Licenses,
                                      ModelVisibility)
from modelscope.hub.errors import (InvalidParameter, NotExistError,
                                   NotLoginException, NoValidRevisionError,
                                   RequestError, datahub_raise_on_error,
                                   handle_http_post_error,
                                   handle_http_response, is_ok,
                                   raise_for_http_status, raise_on_error)
from modelscope.hub.git import GitCommandWrapper
from modelscope.hub.repository import Repository
from modelscope.utils.constant import (DEFAULT_DATASET_REVISION,
                                       DEFAULT_MODEL_REVISION,
                                       DEFAULT_REPOSITORY_REVISION,
                                       MASTER_MODEL_BRANCH, DatasetFormations,
                                       DatasetMetaFormats,
                                       DatasetVisibilityMap, DownloadChannel,
                                       ModelFile)
from modelscope.utils.logger import get_logger
from .utils.utils import (get_endpoint, get_release_datetime,
                          model_id_to_group_owner_name)

logger = get_logger()


class HubApi:
    """Model hub api interface.
    """
    def __init__(self, endpoint: Optional[str] = None):
        """The ModelScope HubApi。

        Args:
            endpoint (str, optional): The modelscope server http|https address. Defaults to None.
        """
        self.endpoint = endpoint if endpoint is not None else get_endpoint()
        self.headers = {'user-agent': ModelScopeConfig.get_user_agent()}
        self.session = Session()
        retry = Retry(
            total=2,
            read=2,
            connect=2,
            backoff_factor=1,
            status_forcelist=(500, 502, 503, 504),
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
                    timeout=API_HTTP_CLIENT_TIMEOUT))

    def login(
        self,
        access_token: str,
    ) -> tuple():
        """Login with your SDK access token, which can be obtained from
           https://www.modelscope.cn user center.

        Args:
            access_token (str): user access token on modelscope.

        Returns:
            cookies: to authenticate yourself to ModelScope open-api
            git_token: token to access your git repository.

        Note:
            You only have to login once within 30 days.
        """
        path = f'{self.endpoint}/api/v1/login'
        r = self.session.post(
            path, json={'AccessToken': access_token}, headers=self.headers)
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
                     chinese_name: Optional[str] = None) -> str:
        """Create model repo at ModelScopeHub.

        Args:
            model_id (str): The model id
            visibility (int, optional): visibility of the model(1-private, 5-public), default 5.
            license (str, optional): license of the model, default none.
            chinese_name (str, optional): chinese name of the model.

        Returns:
            Name of the model created

        Raises:
            InvalidParameter: If model_id is invalid.
            ValueError: If not login.

        Note:
            model_id = {owner}/{name}
        """
        if model_id is None:
            raise InvalidParameter('model_id is required!')
        cookies = ModelScopeConfig.get_cookies()
        if cookies is None:
            raise ValueError('Token does not exist, please login first.')

        path = f'{self.endpoint}/api/v1/models'
        owner_or_group, name = model_id_to_group_owner_name(model_id)
        body = {
            'Path': owner_or_group,
            'Name': name,
            'ChineseName': chinese_name,
            'Visibility': visibility,  # server check
            'License': license
        }
        r = self.session.post(
            path, json=body, cookies=cookies, headers=self.headers)
        handle_http_post_error(r, path, body)
        raise_on_error(r.json())
        model_repo_url = f'{get_endpoint()}/{model_id}'
        return model_repo_url

    def delete_model(self, model_id: str):
        """Delete model_id from ModelScope.

        Args:
            model_id (str): The model id.

        Raises:
            ValueError: If not login.

        Note:
            model_id = {owner}/{name}
        """
        cookies = ModelScopeConfig.get_cookies()
        if cookies is None:
            raise ValueError('Token does not exist, please login first.')
        path = f'{self.endpoint}/api/v1/models/{model_id}'

        r = self.session.delete(path, cookies=cookies, headers=self.headers)
        raise_for_http_status(r)
        raise_on_error(r.json())

    def get_model_url(self, model_id: str):
        return f'{self.endpoint}/api/v1/models/{model_id}.git'

    def get_model(
        self,
        model_id: str,
        revision: Optional[str] = DEFAULT_MODEL_REVISION,
    ) -> str:
        """Get model information at ModelScope

        Args:
            model_id (str): The model id.
            revision (str optional): revision of model.

        Returns:
            The model detail information.

        Raises:
            NotExistError: If the model is not exist, will throw NotExistError

        Note:
            model_id = {owner}/{name}
        """
        cookies = ModelScopeConfig.get_cookies()
        owner_or_group, name = model_id_to_group_owner_name(model_id)
        if revision:
            path = f'{self.endpoint}/api/v1/models/{owner_or_group}/{name}?Revision={revision}'
        else:
            path = f'{self.endpoint}/api/v1/models/{owner_or_group}/{name}'

        r = self.session.get(path, cookies=cookies, headers=self.headers)
        handle_http_response(r, logger, cookies, model_id)
        if r.status_code == HTTPStatus.OK:
            if is_ok(r.json()):
                return r.json()[API_RESPONSE_FIELD_DATA]
            else:
                raise NotExistError(r.json()[API_RESPONSE_FIELD_MESSAGE])
        else:
            raise_for_http_status(r)

    def push_model(self,
                   model_id: str,
                   model_dir: str,
                   visibility: Optional[int] = ModelVisibility.PUBLIC,
                   license: Optional[str] = Licenses.APACHE_V2,
                   chinese_name: Optional[str] = None,
                   commit_message: Optional[str] = 'upload model',
                   revision: Optional[str] = DEFAULT_REPOSITORY_REVISION):
        """Upload model from a given directory to given repository. A valid model directory
        must contain a configuration.json file.

        This function upload the files in given directory to given repository. If the
        given repository is not exists in remote, it will automatically create it with
        given visibility, license and chinese_name parameters. If the revision is also
        not exists in remote repository, it will create a new branch for it.

        This function must be called before calling HubApi's login with a valid token
        which can be obtained from ModelScope's website.

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
            revision (`str`, *optional*, default to DEFAULT_MODEL_REVISION):
                which branch to push. If the branch is not exists, It will create a new
                branch and push to it.

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
            raise ValueError(f'{model_dir} must contain a configuration.json.')
        cookies = ModelScopeConfig.get_cookies()
        if cookies is None:
            raise NotLoginException('Must login before upload!')
        files_to_save = os.listdir(model_dir)
        try:
            self.get_model(model_id=model_id)
        except Exception:
            if visibility is None or license is None:
                raise InvalidParameter(
                    'visibility and license cannot be empty if want to create new repo'
                )
            logger.info('Create new model %s' % model_id)
            self.create_model(
                model_id=model_id,
                visibility=visibility,
                license=license,
                chinese_name=chinese_name)
        tmp_dir = tempfile.mkdtemp()
        git_wrapper = GitCommandWrapper()
        try:
            repo = Repository(model_dir=tmp_dir, clone_from=model_id)
            branches = git_wrapper.get_remote_branches(tmp_dir)
            if revision not in branches:
                logger.info('Create new branch %s' % revision)
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
                    src = os.path.join(model_dir, f)
                    if os.path.isdir(src):
                        shutil.copytree(src, os.path.join(tmp_dir, f))
                    else:
                        shutil.copy(src, tmp_dir)
            if not commit_message:
                date = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
                commit_message = '[automsg] push model %s to hub at %s' % (
                    model_id, date)
            repo.push(
                commit_message=commit_message,
                local_branch=revision,
                remote_branch=revision)
        except Exception:
            raise
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def list_models(self,
                    owner_or_group: str,
                    page_number: Optional[int] = 1,
                    page_size: Optional[int] = 10) -> dict:
        """List models in owner or group.

        Args:
            owner_or_group(str): owner or group.
            page_number(int, optional): The page number, default: 1
            page_size(int, optional): The page size, default: 10

        Raises:
            RequestError: The request error.

        Returns:
            dict: {"models": "list of models", "TotalCount": total_number_of_models_in_owner_or_group}
        """
        cookies = ModelScopeConfig.get_cookies()
        path = f'{self.endpoint}/api/v1/models/'
        r = self.session.put(
            path,
            data='{"Path":"%s", "PageNumber":%s, "PageSize": %s}' %
            (owner_or_group, page_number, page_size),
            cookies=cookies,
            headers=self.headers)
        handle_http_response(r, logger, cookies, 'list_model')
        if r.status_code == HTTPStatus.OK:
            if is_ok(r.json()):
                data = r.json()[API_RESPONSE_FIELD_DATA]
                return data
            else:
                raise RequestError(r.json()[API_RESPONSE_FIELD_MESSAGE])
        else:
            raise_for_http_status(r)
        return None

    def _check_cookie(self,
                      use_cookies: Union[bool,
                                         CookieJar] = False) -> CookieJar:
        cookies = None
        if isinstance(use_cookies, CookieJar):
            cookies = use_cookies
        elif use_cookies:
            cookies = ModelScopeConfig.get_cookies()
            if cookies is None:
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
                                    The timestamp is represented by the seconds elasped from the epoch time.
            use_cookies (Union[bool, CookieJar], optional): If is cookieJar, we will use this cookie, if True, will
                        will load cookie from local. Defaults to False.

        Returns:
            Tuple[List[str], List[str]]: Return list of branch name and tags
        """
        cookies = self._check_cookie(use_cookies)
        if cutoff_timestamp is None:
            cutoff_timestamp = get_release_datetime()
        path = f'{self.endpoint}/api/v1/models/{model_id}/revisions?EndTime=%s' % cutoff_timestamp
        r = self.session.get(path, cookies=cookies, headers=self.headers)
        handle_http_response(r, logger, cookies, model_id)
        d = r.json()
        raise_on_error(d)
        info = d[API_RESPONSE_FIELD_DATA]
        # tags returned from backend are guaranteed to be ordered by create-time
        tags = [x['Revision'] for x in info['RevisionMap']['Tags']
                ] if info['RevisionMap']['Tags'] else []
        return tags

    def get_valid_revision(self,
                           model_id: str,
                           revision=None,
                           cookies: Optional[CookieJar] = None):
        release_timestamp = get_release_datetime()
        current_timestamp = int(round(datetime.datetime.now().timestamp()))
        # for active development in library codes (non-release-branches), release_timestamp
        # is set to be a far-away-time-in-the-future, to ensure that we shall
        # get the master-HEAD version from model repo by default (when no revision is provided)
        if release_timestamp > current_timestamp + ONE_YEAR_SECONDS:
            branches, tags = self.get_model_branches_and_tags(
                model_id, use_cookies=False if cookies is None else cookies)
            if revision is None:
                revision = MASTER_MODEL_BRANCH
                logger.info(
                    'Model revision not specified, use default: %s in development mode'
                    % revision)
            if revision not in branches and revision not in tags:
                raise NotExistError('The model: %s has no revision : %s .' % (model_id, revision))
            logger.info('Development mode use revision: %s' % revision)
        else:
            if revision is None:  # user not specified revision, use latest revision before release time
                revisions = self.list_model_revisions(
                    model_id,
                    cutoff_timestamp=release_timestamp,
                    use_cookies=False if cookies is None else cookies)
                if len(revisions) == 0:
                    raise NoValidRevisionError(
                        'The model: %s has no valid revision!' % model_id)
                # tags (revisions) returned from backend are guaranteed to be ordered by create-time
                # we shall obtain the latest revision created earlier than release version of this branch
                revision = revisions[0]
                logger.info(
                    'Model revision not specified, use the latest revision: %s'
                    % revision)
            else:
                # use user-specified revision
                revisions = self.list_model_revisions(
                    model_id,
                    cutoff_timestamp=current_timestamp,
                    use_cookies=False if cookies is None else cookies)
                if revision not in revisions:
                    raise NotExistError('The model: %s has no revision: %s !' %
                                        (model_id, revision))
                logger.info('Use user-specified model revision: %s' % revision)
        return revision

    def get_model_branches_and_tags(
        self,
        model_id: str,
        use_cookies: Union[bool, CookieJar] = False,
    ) -> Tuple[List[str], List[str]]:
        """Get model branch and tags.

        Args:
            model_id (str): The model id
            use_cookies (Union[bool, CookieJar], optional): If is cookieJar, we will use this cookie, if True, will
                        will load cookie from local. Defaults to False.

        Returns:
            Tuple[List[str], List[str]]: Return list of branch name and tags
        """
        cookies = self._check_cookie(use_cookies)

        path = f'{self.endpoint}/api/v1/models/{model_id}/revisions'
        r = self.session.get(path, cookies=cookies, headers=self.headers)
        handle_http_response(r, logger, cookies, model_id)
        d = r.json()
        raise_on_error(d)
        info = d[API_RESPONSE_FIELD_DATA]
        branches = [x['Revision'] for x in info['RevisionMap']['Branches']
                    ] if info['RevisionMap']['Branches'] else []
        tags = [x['Revision'] for x in info['RevisionMap']['Tags']
                ] if info['RevisionMap']['Tags'] else []
        return branches, tags

    def get_model_files(self,
                        model_id: str,
                        revision: Optional[str] = DEFAULT_MODEL_REVISION,
                        root: Optional[str] = None,
                        recursive: Optional[str] = False,
                        use_cookies: Union[bool, CookieJar] = False,
                        headers: Optional[dict] = {}) -> List[dict]:
        """List the models files.

        Args:
            model_id (str): The model id
            revision (Optional[str], optional): The branch or tag name.
            root (Optional[str], optional): The root path. Defaults to None.
            recursive (Optional[str], optional): Is recursive list files. Defaults to False.
            use_cookies (Union[bool, CookieJar], optional): If is cookieJar, we will use this cookie, if True,
                        will load cookie from local. Defaults to False.
            headers: request headers

        Returns:
            List[dict]: Model file list.
        """
        if revision:
            path = '%s/api/v1/models/%s/repo/files?Revision=%s&Recursive=%s' % (
                self.endpoint, model_id, revision, recursive)
        else:
            path = '%s/api/v1/models/%s/repo/files?Recursive=%s' % (
                self.endpoint, model_id, recursive)
        cookies = self._check_cookie(use_cookies)
        if root is not None:
            path = path + f'&Root={root}'
        headers = self.headers if headers is None else headers
        r = self.session.get(
            path, cookies=cookies, headers=headers)

        handle_http_response(r, logger, cookies, model_id)
        d = r.json()
        raise_on_error(d)

        files = []
        for file in d[API_RESPONSE_FIELD_DATA]['Files']:
            if file['Name'] == '.gitignore' or file['Name'] == '.gitattributes':
                continue

            files.append(file)
        return files

    def list_datasets(self):
        path = f'{self.endpoint}/api/v1/datasets'
        params = {}
        r = self.session.get(path, params=params, headers=self.headers)
        raise_for_http_status(r)
        dataset_list = r.json()[API_RESPONSE_FIELD_DATA]
        return [x['Name'] for x in dataset_list]

    def get_dataset_id_and_type(self, dataset_name: str, namespace: str):
        """ Get the dataset id and type. """
        datahub_url = f'{self.endpoint}/api/v1/datasets/{namespace}/{dataset_name}'
        cookies = ModelScopeConfig.get_cookies()
        r = self.session.get(datahub_url, cookies=cookies)
        resp = r.json()
        datahub_raise_on_error(datahub_url, resp)
        dataset_id = resp['Data']['Id']
        dataset_type = resp['Data']['Type']
        return dataset_id, dataset_type

    def get_dataset_meta_file_list(self, dataset_name: str, namespace: str, dataset_id: str, revision: str):
        """ Get the meta file-list of the dataset. """
        datahub_url = f'{self.endpoint}/api/v1/datasets/{dataset_id}/repo/tree?Revision={revision}'
        cookies = ModelScopeConfig.get_cookies()
        r = self.session.get(datahub_url, cookies=cookies, headers=self.headers)
        r = self.session.get(
            datahub_url, cookies=cookies, headers=self.headers)
        resp = r.json()
        datahub_raise_on_error(datahub_url, resp)
        file_list = resp['Data']
        if file_list is None:
            raise NotExistError(
                f'The modelscope dataset [dataset_name = {dataset_name}, namespace = {namespace}, '
                f'version = {revision}] dose not exist')

        file_list = file_list['Files']
        return file_list

    def get_dataset_meta_files_local_paths(self, dataset_name: str,
                                           namespace: str,
                                           revision: str,
                                           meta_cache_dir: str, dataset_type: int, file_list: list):
        local_paths = defaultdict(list)
        dataset_formation = DatasetFormations(dataset_type)
        dataset_meta_format = DatasetMetaFormats[dataset_formation]
        cookies = ModelScopeConfig.get_cookies()

        # Dump the data_type as a local file
        dataset_type_file_path = os.path.join(meta_cache_dir,
                                              f'{str(dataset_type)}{DatasetFormations.formation_mark_ext.value}')
        with open(dataset_type_file_path, 'w') as fp:
            fp.write('*** Automatically-generated file, do not modify ***')

        for file_info in file_list:
            file_path = file_info['Path']
            extension = os.path.splitext(file_path)[-1]
            if extension in dataset_meta_format:
                datahub_url = f'{self.endpoint}/api/v1/datasets/{namespace}/{dataset_name}/repo?' \
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

    def fetch_single_csv_script(self, script_url: str):
        cookies = ModelScopeConfig.get_cookies()
        resp = self.session.get(script_url, cookies=cookies, headers=self.headers)
        if not resp or not resp.text:
            raise 'The meta-csv file cannot be empty when the meta-args `big_data` is true.'
        text_list = resp.text.strip().split('\n')
        text_headers = text_list[0]
        text_content = text_list[1:]

        return text_headers, text_content

    def get_dataset_file_url(
            self,
            file_name: str,
            dataset_name: str,
            namespace: str,
            revision: Optional[str] = DEFAULT_DATASET_REVISION):
        if file_name.endswith('.csv'):
            file_name = f'{self.endpoint}/api/v1/datasets/{namespace}/{dataset_name}/repo?' \
                        f'Revision={revision}&FilePath={file_name}'
        return file_name

    def get_dataset_access_config(
            self,
            dataset_name: str,
            namespace: str,
            revision: Optional[str] = DEFAULT_DATASET_REVISION):
        datahub_url = f'{self.endpoint}/api/v1/datasets/{namespace}/{dataset_name}/' \
                      f'ststoken?Revision={revision}'
        return self.datahub_remote_call(datahub_url)

    def get_dataset_access_config_session(
            self,
            dataset_name: str,
            namespace: str,
            check_cookie: bool,
            revision: Optional[str] = DEFAULT_DATASET_REVISION):

        datahub_url = f'{self.endpoint}/api/v1/datasets/{namespace}/{dataset_name}/' \
                      f'ststoken?Revision={revision}'
        if check_cookie:
            cookies = self._check_cookie(use_cookies=True)
        else:
            cookies = ModelScopeConfig.get_cookies()
        r = self.session.get(url=datahub_url, cookies=cookies, headers=self.headers)

        r = self.session.get(
            url=datahub_url, cookies=cookies, headers=self.headers)
        resp = r.json()
        raise_on_error(resp)
        return resp['Data']

    def get_dataset_access_config_for_unzipped(self,
                                               dataset_name: str,
                                               namespace: str,
                                               revision: str,
                                               zip_file_name: str):
        datahub_url = f'{self.endpoint}/api/v1/datasets/{namespace}/{dataset_name}'
        cookies = ModelScopeConfig.get_cookies()
        r = self.session.get(url=datahub_url, cookies=cookies, headers=self.headers)
        resp = r.json()
        # get visibility of the dataset
        raise_on_error(resp)
        data = resp['Data']
        visibility = DatasetVisibilityMap.get(data['Visibility'])

        datahub_sts_url = f'{datahub_url}/ststoken?Revision={revision}'
        r_sts = self.session.get(url=datahub_sts_url, cookies=cookies, headers=self.headers)
        resp_sts = r_sts.json()
        raise_on_error(resp_sts)
        data_sts = resp_sts['Data']
        file_dir = visibility + '-unzipped' + '/' + namespace + '_' + dataset_name + '_' + zip_file_name
        data_sts['Dir'] = file_dir
        return data_sts

    def list_oss_dataset_objects(self, dataset_name, namespace, max_limit,
                                 is_recursive, is_filter_dir, revision):
        url = f'{self.endpoint}/api/v1/datasets/{namespace}/{dataset_name}/oss/tree/?' \
            f'MaxLimit={max_limit}&Revision={revision}&Recursive={is_recursive}&FilterDir={is_filter_dir}'

        cookies = ModelScopeConfig.get_cookies()
        resp = self.session.get(url=url, cookies=cookies, timeout=1800)
        resp = resp.json()
        raise_on_error(resp)
        resp = resp['Data']
        return resp

    def delete_oss_dataset_object(self, object_name: str, dataset_name: str,
                                  namespace: str, revision: str) -> str:
        if not object_name or not dataset_name or not namespace or not revision:
            raise ValueError('Args cannot be empty!')

        url = f'{self.endpoint}/api/v1/datasets/{namespace}/{dataset_name}/oss?Path={object_name}&Revision={revision}'

        cookies = ModelScopeConfig.get_cookies()
        resp = self.session.delete(url=url, cookies=cookies)
        resp = resp.json()
        raise_on_error(resp)
        resp = resp['Message']
        return resp

    def delete_oss_dataset_dir(self, object_name: str, dataset_name: str,
                               namespace: str, revision: str) -> str:
        if not object_name or not dataset_name or not namespace or not revision:
            raise ValueError('Args cannot be empty!')

        url = f'{self.endpoint}/api/v1/datasets/{namespace}/{dataset_name}/oss/prefix?Prefix={object_name}/' \
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
        datahub_raise_on_error(url, resp)
        return resp['Data']

    def dataset_download_statistics(self, dataset_name: str, namespace: str, use_streaming: bool) -> None:
        is_ci_test = os.getenv('CI_TEST') == 'True'
        if dataset_name and namespace and not is_ci_test and not use_streaming:
            try:
                cookies = ModelScopeConfig.get_cookies()

                # Download count
                download_count_url = f'{self.endpoint}/api/v1/datasets/{namespace}/{dataset_name}/download/increase'
                download_count_resp = self.session.post(download_count_url, cookies=cookies, headers=self.headers)
                raise_for_http_status(download_count_resp)

                # Download uv
                channel = DownloadChannel.LOCAL.value
                user_name = ''
                if MODELSCOPE_CLOUD_ENVIRONMENT in os.environ:
                    channel = os.environ[MODELSCOPE_CLOUD_ENVIRONMENT]
                if MODELSCOPE_CLOUD_USERNAME in os.environ:
                    user_name = os.environ[MODELSCOPE_CLOUD_USERNAME]
                download_uv_url = f'{self.endpoint}/api/v1/datasets/{namespace}/{dataset_name}/download/uv/' \
                                  f'{channel}?user={user_name}'
                download_uv_resp = self.session.post(download_uv_url, cookies=cookies, headers=self.headers)
                download_uv_resp = download_uv_resp.json()
                raise_on_error(download_uv_resp)

            except Exception as e:
                logger.error(e)


class ModelScopeConfig:
    path_credential = expanduser(DEFAULT_CREDENTIALS_PATH)
    COOKIES_FILE_NAME = 'cookies'
    GIT_TOKEN_FILE_NAME = 'git_token'
    USER_INFO_FILE_NAME = 'user'
    USER_SESSION_ID_FILE_NAME = 'session'

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
                    if cookie.is_expired():
                        logger.warning(
                            'Authentication has expired, '
                            'please re-login if you need to access private models or datasets.')
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
