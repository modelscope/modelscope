import os
import pickle
import subprocess
from http.cookiejar import CookieJar
from os.path import expanduser
from typing import List, Optional, Tuple, Union

import requests

from modelscope.utils.logger import get_logger
from .constants import MODELSCOPE_URL_SCHEME
from .errors import InvalidParameter, NotExistError, is_ok, raise_on_error
from .utils.utils import (get_endpoint, get_gitlab_domain,
                          model_id_to_group_owner_name)

logger = get_logger()


class HubApi:

    def __init__(self, endpoint=None):
        self.endpoint = endpoint if endpoint is not None else get_endpoint()

    def login(
        self,
        user_name: str,
        password: str,
    ) -> tuple():
        """
        Login with username and password

        Args:
            username(`str`): user name on modelscope
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
        model_repo_url = f'{MODELSCOPE_URL_SCHEME}{get_gitlab_domain()}/{model_id}'
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
        path = f'{self.endpoint}/api/v1/models/{model_id}'

        r = requests.delete(path, cookies=cookies)
        r.raise_for_status()
        raise_on_error(r.json())

    def get_model_url(self, model_id):
        return f'{self.endpoint}/api/v1/models/{model_id}.git'

    def get_model(
        self,
        model_id: str,
        revision: str = 'master',
    ) -> str:
        """
        Get model information at modelscope_hub

        Args:
            model_id(`str`): The model id.
            revision(`str`): revision of model
        Returns:
            The model details information.
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
        r.raise_for_status()
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
                        revision: Optional[str] = 'master',
                        root: Optional[str] = None,
                        recursive: Optional[str] = False,
                        use_cookies: Union[bool, CookieJar] = False,
                        headers: Optional[dict] = {}) -> List[dict]:
        """List the models files.

        Args:
            model_id (str): The model id
            revision (Optional[str], optional): The branch or tag name. Defaults to 'master'.
            root (Optional[str], optional): The root path. Defaults to None.
            recursive (Optional[str], optional): Is recurive list files. Defaults to False.
            use_cookies (Union[bool, CookieJar], optional): If is cookieJar, we will use this cookie, if True, will
                        will load cookie from local. Defaults to False.
            is_snapshot(Optional[bool], optional): when snapshot_download set to True, otherwise False.

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

        r.raise_for_status()
        d = r.json()
        raise_on_error(d)

        files = []
        for file in d['Data']['Files']:
            if file['Name'] == '.gitignore' or file['Name'] == '.gitattributes':
                continue

            files.append(file)
        return files


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
        try:
            cookies_path = os.path.join(cls.path_credential, 'cookies')
            with open(cookies_path, 'rb') as f:
                cookies = pickle.load(f)
                for cookie in cookies:
                    if cookie.is_expired():
                        logger.warn('Auth is expored, please re-login')
                        return None
                return cookies
        except FileNotFoundError:
            logger.warn(
                "Auth token does not exist, you'll get authentication error when downloading \
                private model files. Please login first")
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
