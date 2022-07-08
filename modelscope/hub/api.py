import os
import pickle
import shutil
import subprocess
from collections import defaultdict
from http.cookiejar import CookieJar
from os.path import expanduser
from typing import List, Optional, Tuple, Union

import requests

from modelscope.utils.logger import get_logger
from ..msdatasets.config import DOWNLOADED_DATASETS_PATH, HUB_DATASET_ENDPOINT
from ..utils.constant import (DEFAULT_DATASET_REVISION, DEFAULT_MODEL_REVISION,
                              DownloadMode)
from .errors import (InvalidParameter, NotExistError, datahub_raise_on_error,
                     handle_http_response, is_ok, raise_on_error)
from .utils.utils import get_endpoint, model_id_to_group_owner_name

logger = get_logger()


class HubApi:

    def __init__(self, endpoint=None, dataset_endpoint=None):
        self.endpoint = endpoint if endpoint is not None else get_endpoint()
        self.dataset_endpoint = dataset_endpoint if dataset_endpoint is not None else HUB_DATASET_ENDPOINT

    def login(
        self,
        user_name: str,
        password: str,
    ) -> tuple():
        """
        Login with username and password

        Args:
            user_name(`str`): user name on modelscope
            password(`str`): password

        Returns:
            cookies: to authenticate yourself to ModelScope open-api
            gitlab token: to access private repos

        <Tip>
            You only have to login once within 30 days.
        </Tip>
        """
        path = f'{self.endpoint}/api/v1/login'
        r = requests.post(
            path, json={
                'username': user_name,
                'password': password
            })
        r.raise_for_status()
        d = r.json()
        raise_on_error(d)

        token = d['Data']['AccessToken']
        cookies = r.cookies

        # save token and cookie
        ModelScopeConfig.save_token(token)
        ModelScopeConfig.save_cookies(cookies)
        ModelScopeConfig.write_to_git_credential(user_name, password)

        return d['Data']['AccessToken'], cookies

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
        path = f'{self.endpoint}/api/v1/models/{owner_or_group}/{name}?{revision}'

        r = requests.get(path, cookies=cookies)
        handle_http_response(r, logger, cookies, model_id)
        if r.status_code == 200:
            if is_ok(r.json()):
                return r.json()['Data']
            else:
                raise NotExistError(r.json()['Message'])
        else:
            r.raise_for_status()

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
            Tuple[List[str], List[str]]: _description_
        """
        cookies = self._check_cookie(use_cookies)

        path = f'{self.endpoint}/api/v1/models/{model_id}/revisions'
        r = requests.get(path, cookies=cookies)
        handle_http_response(r, logger, cookies, model_id)
        d = r.json()
        raise_on_error(d)
        info = d['Data']
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
        for file in d['Data']['Files']:
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
        dataset_list = r.json()['Data']
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
        cache_dir = os.path.join(DOWNLOADED_DATASETS_PATH, dataset_name,
                                 namespace, revision)
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
        for file_info in file_list:
            file_path = file_info['Path']
            if file_path.endswith('.py'):
                datahub_url = f'{self.dataset_endpoint}/api/v1/datasets/{dataset_id}/repo/files?' \
                              f'Revision={revision}&Path={file_path}'
                r = requests.get(datahub_url)
                r.raise_for_status()
                content = r.json()['Data']['Content']
                local_path = os.path.join(cache_dir, file_path)
                if os.path.exists(local_path):
                    logger.warning(
                        f"Reusing dataset {dataset_name}'s python file ({local_path})"
                    )
                    local_paths['py'].append(local_path)
                    continue
                with open(local_path, 'w') as f:
                    f.writelines(content)
                local_paths['py'].append(local_path)
        return local_paths


class ModelScopeConfig:
    path_credential = expanduser('~/.modelscope/credentials')

    @classmethod
    def make_sure_credential_path_exist(cls):
        os.makedirs(cls.path_credential, exist_ok=True)

    @classmethod
    def save_cookies(cls, cookies: CookieJar):
        cls.make_sure_credential_path_exist()
        with open(os.path.join(cls.path_credential, 'cookies'), 'wb+') as f:
            pickle.dump(cookies, f)

    @classmethod
    def get_cookies(cls):
        cookies_path = os.path.join(cls.path_credential, 'cookies')
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

    @classmethod
    def save_token(cls, token: str):
        cls.make_sure_credential_path_exist()
        with open(os.path.join(cls.path_credential, 'token'), 'w+') as f:
            f.write(token)

    @classmethod
    def get_token(cls) -> Optional[str]:
        """
        Get token or None if not existent.

        Returns:
            `str` or `None`: The token, `None` if it doesn't exist.

        """
        token = None
        try:
            with open(os.path.join(cls.path_credential, 'token'), 'r') as f:
                token = f.read()
        except FileNotFoundError:
            pass
        return token

    @staticmethod
    def write_to_git_credential(username: str, password: str):
        with subprocess.Popen(
                'git credential-store store'.split(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
        ) as process:
            input_username = f'username={username.lower()}'
            input_password = f'password={password}'

            process.stdin.write(
                f'url={get_endpoint()}\n{input_username}\n{input_password}\n\n'
                .encode('utf-8'))
            process.stdin.flush()
