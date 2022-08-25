from http.cookiejar import CookieJar

from .oss_utils import OssUtilities


class DatasetUploadManager(object):

    def __init__(self, dataset_name: str, namespace: str, version: str,
                 cookies: CookieJar):
        from modelscope.hub.api import HubApi
        api = HubApi()
        oss_config = api.get_dataset_access_config_session(
            cookies=cookies,
            dataset_name=dataset_name,
            namespace=namespace,
            revision=version)

        self.oss_utilities = OssUtilities(oss_config)

    def upload(self, oss_file_name: str, local_file_path: str) -> str:
        oss_object_key = self.oss_utilities.upload(
            oss_file_name=oss_file_name, local_file_path=local_file_path)
        return oss_object_key
