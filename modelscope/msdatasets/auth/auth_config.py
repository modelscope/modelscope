# Copyright (c) Alibaba, Inc. and its affiliates.

from http.cookiejar import CookieJar
from typing import Tuple


class BaseAuthConfig(object):
    """Base authorization config class."""

    def __init__(self, cookies: CookieJar, git_token: str,
                 user_info: Tuple[str, str]):
        self.cookies = cookies
        self.git_token = git_token
        self.user_info = user_info


class OssAuthConfig(BaseAuthConfig):
    """The authorization config for oss dataset."""

    def __init__(self, cookies: CookieJar, git_token: str,
                 user_info: Tuple[str, str]):
        super().__init__(
            cookies=cookies, git_token=git_token, user_info=user_info)


class VirgoAuthConfig(BaseAuthConfig):
    """The authorization config for virgo dataset."""

    def __init__(self, cookies: CookieJar, git_token: str,
                 user_info: Tuple[str, str]):
        super().__init__(
            cookies=cookies, git_token=git_token, user_info=user_info)


class MaxComputeAuthConfig(BaseAuthConfig):
    # TODO: MaxCompute dataset to be supported.
    def __init__(self, cookies: CookieJar, git_token: str,
                 user_info: Tuple[str, str]):
        super().__init__(
            cookies=cookies, git_token=git_token, user_info=user_info)

        self.max_compute_grant_cmd = None
