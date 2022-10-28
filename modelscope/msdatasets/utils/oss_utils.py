# Copyright (c) Alibaba, Inc. and its affiliates.

from __future__ import print_function
import os

import oss2
from datasets.utils.file_utils import hash_url_to_filename

from modelscope.hub.api import HubApi
from modelscope.utils.constant import UploadMode
from modelscope.utils.logger import get_logger

logger = get_logger()

ACCESS_ID = 'AccessId'
ACCESS_SECRET = 'AccessSecret'
SECURITY_TOKEN = 'SecurityToken'
BUCKET = 'Bucket'
BACK_DIR = 'BackupDir'
DIR = 'Dir'


class OssUtilities:

    def __init__(self, oss_config, dataset_name, namespace, revision):
        self._do_init(oss_config=oss_config)

        self.dataset_name = dataset_name
        self.namespace = namespace
        self.revision = revision

        self.upload_resumable_tmp_store = '/tmp/modelscope/tmp_dataset'
        self.upload_multipart_threshold = 50 * 1024 * 1024
        self.upload_part_size = 1 * 1024 * 1024
        self.upload_num_threads = 4
        self.upload_max_retries = 3

        self.api = HubApi()

    def _do_init(self, oss_config):
        self.key = oss_config[ACCESS_ID]
        self.secret = oss_config[ACCESS_SECRET]
        self.token = oss_config[SECURITY_TOKEN]
        self.endpoint = f"https://{oss_config['Region']}.aliyuncs.com"
        self.bucket_name = oss_config[BUCKET]
        auth = oss2.StsAuth(self.key, self.secret, self.token)
        self.bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)
        self.oss_dir = oss_config[DIR]
        self.oss_backup_dir = oss_config[BACK_DIR]

    def _reload_sts(self):
        cookies = self.api.check_local_cookies(use_cookies=True)
        oss_config_refresh = self.api.get_dataset_access_config_session(
            cookies=cookies,
            dataset_name=self.dataset_name,
            namespace=self.namespace,
            revision=self.revision)
        self._do_init(oss_config_refresh)

    @staticmethod
    def _percentage(consumed_bytes, total_bytes):
        if total_bytes:
            rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
            print('\r{0}% '.format(rate), end='', flush=True)

    def download(self, oss_file_name, download_config):
        cache_dir = download_config.cache_dir
        candidate_key = os.path.join(self.oss_dir, oss_file_name)
        candidate_key_backup = os.path.join(self.oss_backup_dir, oss_file_name)
        file_oss_key = candidate_key if self.bucket.object_exists(
            candidate_key) else candidate_key_backup
        filename = hash_url_to_filename(file_oss_key, etag=None)
        local_path = os.path.join(cache_dir, filename)

        if download_config.force_download or not os.path.exists(local_path):
            oss2.resumable_download(
                self.bucket,
                file_oss_key,
                local_path,
                multiget_threshold=0,
                progress_callback=self._percentage)
        return local_path

    def upload(self, oss_object_name: str, local_file_path: str,
               indicate_individual_progress: bool,
               upload_mode: UploadMode) -> str:
        retry_count = 0
        object_key = os.path.join(self.oss_dir, oss_object_name)
        resumable_store = oss2.ResumableStore(
            root=self.upload_resumable_tmp_store)
        if indicate_individual_progress:
            progress_callback = self._percentage
        else:
            progress_callback = None

        while True:
            try:
                retry_count += 1
                exist = self.bucket.object_exists(object_key)
                if upload_mode == UploadMode.APPEND and exist:
                    logger.info(
                        f'Skip {oss_object_name} in case of {upload_mode.value} mode.'
                    )
                    break

                oss2.resumable_upload(
                    self.bucket,
                    object_key,
                    local_file_path,
                    store=resumable_store,
                    multipart_threshold=self.upload_multipart_threshold,
                    part_size=self.upload_part_size,
                    progress_callback=progress_callback,
                    num_threads=self.upload_num_threads)
                break
            except Exception as e:
                if e.__getattribute__('status') == 403:
                    self._reload_sts()
                if retry_count >= self.upload_max_retries:
                    raise

        return object_key
