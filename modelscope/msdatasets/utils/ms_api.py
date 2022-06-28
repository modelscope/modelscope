import os
import shutil
from collections import defaultdict
from typing import Optional

import requests

from modelscope.hub.errors import NotExistError, datahub_raise_on_error
from modelscope.msdatasets.config import (DOWNLOADED_DATASETS_PATH,
                                          MS_HUB_ENDPOINT)
from modelscope.utils.constant import DownloadMode
from modelscope.utils.logger import get_logger

logger = get_logger()


class MsApi:

    def __init__(self, endpoint=MS_HUB_ENDPOINT):
        self.endpoint = endpoint

    def list_datasets(self):
        path = f'{self.endpoint}/api/v1/datasets'
        headers = None
        params = {}
        r = requests.get(path, params=params, headers=headers)
        r.raise_for_status()
        dataset_list = r.json()['Data']
        return [x['Name'] for x in dataset_list]

    def fetch_dataset_scripts(self,
                              dataset_name: str,
                              namespace: str,
                              download_mode: Optional[DownloadMode],
                              version: Optional[str] = 'master'):
        if namespace is None:
            raise ValueError(
                f'Dataset from Hubs.modelscope should have a valid "namespace", but get {namespace}'
            )
        version = version or 'master'
        cache_dir = os.path.join(DOWNLOADED_DATASETS_PATH, dataset_name,
                                 namespace, version)
        download_mode = DownloadMode(download_mode
                                     or DownloadMode.REUSE_DATASET_IF_EXISTS)
        if download_mode == DownloadMode.FORCE_REDOWNLOAD and os.path.exists(
                cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        datahub_url = f'{self.endpoint}/api/v1/datasets/{namespace}/{dataset_name}'
        r = requests.get(datahub_url)
        resp = r.json()
        datahub_raise_on_error(datahub_url, resp)
        dataset_id = resp['Data']['Id']
        datahub_url = f'{self.endpoint}/api/v1/datasets/{dataset_id}/repo/tree?Revision={version}'
        r = requests.get(datahub_url)
        resp = r.json()
        datahub_raise_on_error(datahub_url, resp)
        file_list = resp['Data']
        if file_list is None:
            raise NotExistError(
                f'The modelscope dataset [dataset_name = {dataset_name}, namespace = {namespace}, '
                f'version = {version}] dose not exist')

        file_list = file_list['Files']
        local_paths = defaultdict(list)
        for file_info in file_list:
            file_path = file_info['Path']
            if file_path.endswith('.py'):
                datahub_url = f'{self.endpoint}/api/v1/datasets/{dataset_id}/repo/files?' \
                              f'Revision={version}&Path={file_path}'
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
