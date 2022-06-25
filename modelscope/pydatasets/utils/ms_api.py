import os
from collections import defaultdict
from typing import Optional

import requests

from modelscope.pydatasets.config import (DOWNLOADED_DATASETS_PATH,
                                          MS_HUB_ENDPOINT)
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
                              version: Optional[str] = 'master',
                              force_download=False):
        datahub_url = f'{self.endpoint}/api/v1/datasets?Query={dataset_name}'
        r = requests.get(datahub_url)
        r.raise_for_status()
        dataset_list = r.json()['Data']
        if len(dataset_list) == 0:
            return None
        dataset_id = dataset_list[0]['Id']
        version = version or 'master'
        datahub_url = f'{self.endpoint}/api/v1/datasets/{dataset_id}/repo/tree?Revision={version}'
        r = requests.get(datahub_url)
        r.raise_for_status()
        file_list = r.json()['Data']['Files']
        cache_dir = os.path.join(DOWNLOADED_DATASETS_PATH, dataset_name,
                                 version)
        os.makedirs(cache_dir, exist_ok=True)
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
                if os.path.exists(local_path) and not force_download:
                    logger.warning(
                        f"Reusing dataset {dataset_name}'s python file ({local_path})"
                    )
                    local_paths['py'].append(local_path)
                    continue
                with open(local_path, 'w') as f:
                    f.writelines(content)
                local_paths['py'].append(local_path)
        return local_paths
