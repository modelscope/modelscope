# Copyright (c) Alibaba, Inc. and its affiliates.

from .oss_utils import OssUtilities


class DatasetUploadManager(object):

    def __init__(self, dataset_name: str, namespace: str, version: str):
        from modelscope.hub.api import HubApi
        _hub_api = HubApi()
        _cookies = _hub_api.check_cookies_upload_data(use_cookies=True)
        _oss_config = _hub_api.get_dataset_access_config_session(
            cookies=_cookies,
            dataset_name=dataset_name,
            namespace=namespace,
            revision=version)

        self.oss_utilities = OssUtilities(_oss_config)

    def upload(self, object_name: str, local_file_path: str) -> str:
        object_key = self.oss_utilities.upload(
            oss_object_name=object_name, local_file_path=local_file_path)
        return object_key
