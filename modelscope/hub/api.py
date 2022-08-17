import os
import pickle
import shutil
import subprocess
from collections import defaultdict
from http import HTTPStatus
from http.cookiejar import CookieJar
from os.path import expanduser
from typing import List, Optional, Tuple, Union

import requests

from modelscope.hub.constants import (API_RESPONSE_FIELD_DATA,
                                      API_RESPONSE_FIELD_EMAIL,
                                      API_RESPONSE_FIELD_GIT_ACCESS_TOKEN,
                                      API_RESPONSE_FIELD_MESSAGE,
                                      API_RESPONSE_FIELD_USERNAME,
                                      DEFAULT_CREDENTIALS_PATH)
from modelscope.msdatasets.config import (DOWNLOADED_DATASETS_PATH,
                                          HUB_DATASET_ENDPOINT)
from modelscope.utils.constant import (DEFAULT_DATASET_REVISION,
                                       DEFAULT_MODEL_REVISION,
                                       DatasetFormations, DatasetMetaFormats,
                                       DownloadMode)
from modelscope.utils.logger import get_logger
from .errors import (InvalidParameter, NotExistError, RequestError,
                     datahub_raise_on_error, handle_http_response, is_ok,
                     raise_on_error)
from .utils.utils import get_endpoint, model_id_to_group_owner_name

logger = get_logger()


class HubApi:

    def __init__(self, endpoint=None, dataset_endpoint=None):
        self.endpoint = endpoint if endpoint is not None else get_endpoint()
        self.dataset_endpoint = dataset_endpoint if dataset_endpoint is not None else HUB_DATASET_ENDPOINT

    def login(
        self,
        access_token: str,
    ) -> tuple():
        """
        Login with username and password

        Args:
            access_token(`str`): user access token on modelscope.
        Returns:
            cookies: to authenticate yourself to ModelScope open-api
            gitlab token: to access private repos

        <Tip>
            You only have to login once within 30 days.
        </Tip>
        """
        path = f'{self.endpoint}/api/v1/login'
        r = requests.post(path, json={'AccessToken': access_token})
        r.raise_for_status()
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

    def create_model(
        self,
        model_id: str,
        visibility: str,
        license: str,
        chinese_name: Optional[str] = None,
    ) -> str:
        """
        Create model repo at ModelScopeHub

        Args:
            model_id:(`str`): The model id
            visibility(`int`): visibility of the model(1-private, 5-public), default public.
            license(`str`): license of the model, default none.
            chinese_name(`str`, *optional*): chinese name of the model
        Returns:
            name of the model created

        <Tip>
            model_id = {owner}/{name}
        </Tip>
        """
        if model_id is None:
            raise InvalidParameter('model_id is required!')
        cookies = ModelScopeConfig.get_cookies()
        if cookies is None:
            raise ValueError('Token does not exist, please login first.')

        path = f'{self.endpoint}/api/v1/models'
        owner_or_group, name = model_id_to_group_owner_name(model_id)
        r = requests.post(
            path,
            json={
                'Path': owner_or_group,
                'Name': name,
                'ChineseName': chinese_name,
                'Visibility': visibility,  # server check
                'License': license
            },
            cookies=cookies)
        r.raise_for_status()
        raise_on_error(r.json())
        model_repo_url = f'{get_endpoint()}/{model_id}'
        return model_repo_url

    def delete_model(self, model_id):
        """_summary_

        Args:
            model_id (str): The model id.
        <Tip>
            model_id = {owner}/{name}
        </Tip>
        """
        cookies = ModelScopeConfig.get_cookies()
        if cookies is None:
            raise ValueError('Token does not exist, please login first.')
        path = f'{self.endpoint}/api/v1/models/{model_id}'

        r = requests.delete(path, cookies=cookies)
        r.raise_for_status()
        raise_on_error(r.json())

    def get_model_url(self, model_id):
        return f'{self.endpoint}/api/v1/models/{model_id}.git'

    def get_model(
        self,
        model_id: str,
        revision: str = DEFAULT_MODEL_REVISION,
    ) -> str:
        """
        Get model information at modelscope_hub

        Args:
            model_id(`str`): The model id.
            revision(`str`): revision of model
        Returns:
            The model detail information.
        Raises:
            NotExistError: If the model is not exist, will throw NotExistError
        <Tip>
            model_id = {owner}/{name}
        </Tip>
        """
        cookies = ModelScopeConfig.get_cookies()
        owner_or_group, name = model_id_to_group_owner_name(model_id)
        path = f'{self.endpoint}/api/v1/models/{owner_or_group}/{name}?Revision={revision}'

        r = requests.get(path, cookies=cookies)
        handle_http_response(r, logger, cookies, model_id)
        if r.status_code == HTTPStatus.OK:
            if is_ok(r.json()):
                return r.json()[API_RESPONSE_FIELD_DATA]
            else:
                raise NotExistError(r.json()[API_RESPONSE_FIELD_MESSAGE])
        else:
            r.raise_for_status()

    def list_model(self,
                   owner_or_group: str,
                   page_number=1,
                   page_size=10) -> dict:
        """List model in owner or group.

        Args:
            owner_or_group(`str`): owner or group.
            page_number(`int`): The page number, default: 1
            page_size(`int`): The page size, default: 10
        Returns:
            dict: {"models": "list of models", "TotalCount": total_number_of_models_in_owner_or_group}
        """
        cookies = ModelScopeConfig.get_cookies()
        path = f'{self.endpoint}/api/v1/models/'
        r = requests.put(
            path,
            data='{"Path":"%s", "PageNumber":%s, "PageSize": %s}' %
            (owner_or_group, page_number, page_size),
            cookies=cookies)
        handle_http_response(r, logger, cookies, 'list_model')
        if r.status_code == HTTPStatus.OK:
            if is_ok(r.json()):
                data = r.json()[API_RESPONSE_FIELD_DATA]
                return data
            else:
                raise RequestError(r.json()[API_RESPONSE_FIELD_MESSAGE])
        else:
            r.raise_for_status()
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

    def get_model_branches_and_tags(
        self,
        model_id: str,
        use_cookies: Union[bool, CookieJar] = False
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
        r = requests.get(path, cookies=cookies)
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

        Raises:
            ValueError: If user_cookies is True, but no local cookie.

        Returns:
            List[dict]: Model file list.
        """
        path = '%s/api/v1/models/%s/repo/files?Revision=%s&Recursive=%s' % (
            self.endpoint, model_id, revision, recursive)
        cookies = self._check_cookie(use_cookies)
        if root is not None:
            path = path + f'&Root={root}'

        r = requests.get(path, cookies=cookies, headers=headers)

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
        path = f'{self.dataset_endpoint}/api/v1/datasets'
        headers = None
        params = {}
        r = requests.get(path, params=params, headers=headers)
        r.raise_for_status()
        dataset_list = r.json()[API_RESPONSE_FIELD_DATA]
        return [x['Name'] for x in dataset_list]

    def fetch_dataset_scripts(
            self,
            dataset_name: str,
            namespace: str,
            download_mode: Optional[DownloadMode],
            revision: Optional[str] = DEFAULT_DATASET_REVISION):
        if namespace is None:
            raise ValueError(
                f'Dataset from Hubs.modelscope should have a valid "namespace", but get {namespace}'
            )
        revision = revision or DEFAULT_DATASET_REVISION
        cache_dir = os.path.join(DOWNLOADED_DATASETS_PATH, namespace,
                                 dataset_name, revision)
        download_mode = DownloadMode(download_mode
                                     or DownloadMode.REUSE_DATASET_IF_EXISTS)
        if download_mode == DownloadMode.FORCE_REDOWNLOAD and os.path.exists(
                cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        datahub_url = f'{self.dataset_endpoint}/api/v1/datasets/{namespace}/{dataset_name}'
        r = requests.get(datahub_url)
        resp = r.json()
        datahub_raise_on_error(datahub_url, resp)
        dataset_id = resp['Data']['Id']
        dataset_type = resp['Data']['Type']
        datahub_url = f'{self.dataset_endpoint}/api/v1/datasets/{dataset_id}/repo/tree?Revision={revision}'
        r = requests.get(datahub_url)
        resp = r.json()
        datahub_raise_on_error(datahub_url, resp)
        file_list = resp['Data']
        if file_list is None:
            raise NotExistError(
                f'The modelscope dataset [dataset_name = {dataset_name}, namespace = {namespace}, '
                f'version = {revision}] dose not exist')

        file_list = file_list['Files']
        local_paths = defaultdict(list)
        dataset_formation = DatasetFormations(dataset_type)
        dataset_meta_format = DatasetMetaFormats[dataset_formation]
        for file_info in file_list:
            file_path = file_info['Path']
            extension = os.path.splitext(file_path)[-1]
            if extension in dataset_meta_format:
                datahub_url = f'{self.dataset_endpoint}/api/v1/datasets/{namespace}/{dataset_name}/repo?' \
                              f'Revision={revision}&FilePath={file_path}'
                r = requests.get(datahub_url)
                r.raise_for_status()
                local_path = os.path.join(cache_dir, file_path)
                if os.path.exists(local_path):
                    logger.warning(
                        f"Reusing dataset {dataset_name}'s python file ({local_path})"
                    )
                    local_paths[extension].append(local_path)
                    continue
                with open(local_path, 'wb') as f:
                    f.write(r.content)
                local_paths[extension].append(local_path)

        return local_paths, dataset_formation, cache_dir

    def get_dataset_file_url(
            self,
            file_name: str,
            dataset_name: str,
            namespace: str,
            revision: Optional[str] = DEFAULT_DATASET_REVISION):
        if file_name.endswith('.csv'):
            file_name = f'{self.dataset_endpoint}/api/v1/datasets/{namespace}/{dataset_name}/repo?' \
                        f'Revision={revision}&FilePath={file_name}'
        return file_name

    def get_dataset_access_config(
            self,
            dataset_name: str,
            namespace: str,
            revision: Optional[str] = DEFAULT_DATASET_REVISION):
        datahub_url = f'{self.dataset_endpoint}/api/v1/datasets/{namespace}/{dataset_name}/' \
                      f'ststoken?Revision={revision}'
        return self.datahub_remote_call(datahub_url)

    @staticmethod
    def datahub_remote_call(url):
        r = requests.get(url)
        resp = r.json()
        datahub_raise_on_error(url, resp)
        return resp['Data']


class ModelScopeConfig:
    path_credential = expanduser(DEFAULT_CREDENTIALS_PATH)
    COOKIES_FILE_NAME = 'cookies'
    GIT_TOKEN_FILE_NAME = 'git_token'
    USER_INFO_FILE_NAME = 'user'

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
                        logger.warn(
                            'Authentication has expired, please re-login')
                        return None
                return cookies
        return None

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
                    'r') as f:
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
                    'r') as f:
                token = f.read()
        except FileNotFoundError:
            pass
        return token
