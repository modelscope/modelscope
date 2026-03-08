# Copyright (c) Alibaba, Inc. and its affiliates.

from __future__ import print_function
import multiprocessing
import os
import threading

import oss2
from datasets.utils.file_utils import hash_url_to_filename
from oss2 import CredentialsProvider
from oss2.credentials import Credentials

from modelscope.hub.api import HubApi
from modelscope.msdatasets.download.download_config import DataDownloadConfig
from modelscope.utils.config_ds import MS_CACHE_HOME
from modelscope.utils.constant import (DEFAULT_DATA_ACCELERATION_ENDPOINT,
                                       MetaDataFields, UploadMode)
from modelscope.utils.logger import get_logger

logger = get_logger()

ACCESS_ID = 'AccessId'
ACCESS_SECRET = 'AccessSecret'
SECURITY_TOKEN = 'SecurityToken'
BUCKET = 'Bucket'
BACK_DIR = 'BackupDir'
DIR = 'Dir'


class CredentialProviderWrapper(CredentialsProvider):
    """
    A custom credentials provider for oss2 that fetches temporary credentials
    """

    def __init__(self, api: HubApi, dataset_name: str, namespace: str,
                 revision: str):
        """
        Initializes the CredentialProviderWrapper with dataset information.

        Args:
            dataset_name (str): The name of the dataset.
            namespace (str): The namespace of the dataset.
            revision (str): The revision of the dataset.
        """
        self.api = api
        self.dataset_name = dataset_name
        self.namespace = namespace
        self.revision = revision
        self._lock = threading.Lock()

    def get_credentials(self):
        """
        oss2 SDK will call this method automatically when it finds the token is expired or needs authentication.
        """
        with self._lock:
            oss_config = self.api.get_dataset_access_config_session(
                dataset_name=self.dataset_name,
                namespace=self.namespace,
                check_cookie=False,
                revision=self.revision)

            return Credentials(
                access_key_id=oss_config[ACCESS_ID],
                access_key_secret=oss_config[ACCESS_SECRET],
                security_token=oss_config[SECURITY_TOKEN],
            )


class OssUtilities:
    """
    A utility class for handling Alibaba Cloud OSS operations such as upload and download.
    """

    def __init__(self, dataset_name, namespace, revision):
        """
        Initializes the OssUtilities with the given OSS configuration and dataset information.
        """
        self.dataset_name = dataset_name
        self.namespace = namespace
        self.revision = revision

        self.api = HubApi()
        oss_config = self.api.get_dataset_access_config_session(
            dataset_name=self.dataset_name,
            namespace=self.namespace,
            check_cookie=False,
            revision=self.revision)

        if os.getenv('ENABLE_DATASET_ACCELERATION') == 'True':
            self.endpoint = DEFAULT_DATA_ACCELERATION_ENDPOINT
        else:
            self.endpoint = f"https://{oss_config['Region']}.aliyuncs.com"

        self.resumable_store_root_path = os.path.join(MS_CACHE_HOME,
                                                      'tmp/resumable_store')
        self.num_threads = multiprocessing.cpu_count()
        self.part_size = 1 * 1024 * 1024
        self.multipart_threshold = 50 * 1024 * 1024
        self.max_retries = 3

        self.resumable_store_download = oss2.ResumableDownloadStore(
            root=self.resumable_store_root_path)
        self.resumable_store_upload = oss2.ResumableStore(
            root=self.resumable_store_root_path)

        credential_provider = CredentialProviderWrapper(
            api=self.api,
            dataset_name=self.dataset_name,
            namespace=self.namespace,
            revision=self.revision)
        auth = oss2.ProviderAuthV4(credential_provider)

        self.bucket_name = oss_config[BUCKET]
        self.bucket = oss2.Bucket(
            auth=auth,
            endpoint=self.endpoint,
            bucket_name=self.bucket_name,
            region=oss_config['Region'].lstrip('oss-'),
        )
        self.oss_dir = oss_config[DIR]
        self.oss_backup_dir = oss_config[BACK_DIR]

    @staticmethod
    def _percentage(consumed_bytes, total_bytes):
        if total_bytes:
            rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
            print('\r{0}% '.format(rate), end='', flush=True)

    def download(self, oss_file_name: str,
                 download_config: DataDownloadConfig) -> str:
        """
        Downloads a file from OSS to the local cache.

        Args:
            oss_file_name (str): The name of the file in OSS to download.
            download_config (DataDownloadConfig): Configuration for the download process.

        Returns:
            str: The local path to the downloaded file.
        """
        cache_dir = download_config.cache_dir
        candidate_key = os.path.join(self.oss_dir, oss_file_name)
        candidate_key_backup = os.path.join(self.oss_backup_dir, oss_file_name)
        split = download_config.split

        big_data = False
        if split:
            args_dict = download_config.meta_args_map.get(split)
            if args_dict:
                big_data = args_dict.get(MetaDataFields.ARGS_BIG_DATA)

        retry_count = 0
        while True:
            try:
                # big_data is True when the dataset contains large number of objects
                if big_data:
                    file_oss_key = candidate_key
                else:
                    file_oss_key = candidate_key if self.bucket.object_exists(
                        candidate_key) else candidate_key_backup
                filename = hash_url_to_filename(file_oss_key, etag=None)
                local_path = os.path.join(cache_dir, filename)

                if download_config.force_download or not os.path.exists(
                        local_path):
                    oss2.resumable_download(
                        bucket=self.bucket,
                        key=file_oss_key,
                        filename=local_path,
                        store=self.resumable_store_download,
                        multiget_threshold=self.multipart_threshold,
                        part_size=self.part_size,
                        progress_callback=self._percentage,
                        num_threads=self.num_threads)
                break
            except Exception as e:
                logger.warning(
                    f'Error downloading {oss_file_name}: {e}, trying again...')
                retry_count += 1
                if retry_count >= self.max_retries:
                    logger.error(
                        f'Failed to download {oss_file_name} due to exceeded retries.'
                    )
                    raise e

        return local_path

    def upload(self, oss_object_name: str, local_file_path: str,
               indicate_individual_progress: bool,
               upload_mode: UploadMode) -> str:
        """
        Uploads a local file to OSS.

        Args:
            oss_object_name (str): The name of the object in OSS.
            local_file_path (str): The local file path to upload.
            indicate_individual_progress (bool): Whether to show individual progress.
            upload_mode (UploadMode): The upload mode (e.g., OVERWRITE, APPEND).

        Returns:
            str: The OSS object key where the file is uploaded.
        """
        retry_count = 0
        object_key = os.path.join(self.oss_dir, oss_object_name)

        if indicate_individual_progress:
            progress_callback = self._percentage
        else:
            progress_callback = None

        while True:
            try:
                exist = self.bucket.object_exists(object_key)
                if upload_mode == UploadMode.APPEND and exist:
                    logger.info(
                        f'Skip {oss_object_name} in case of {upload_mode.value} mode.'
                    )
                    break

                oss2.resumable_upload(
                    bucket=self.bucket,
                    key=object_key,
                    filename=local_file_path,
                    store=self.resumable_store_upload,
                    multipart_threshold=self.multipart_threshold,
                    part_size=self.part_size,
                    progress_callback=progress_callback,
                    num_threads=self.num_threads)
                break
            except Exception as e:
                logger.warning(
                    f'Error uploading {oss_object_name}: {e}, trying again...')
                retry_count += 1
                if retry_count >= self.max_retries:
                    logger.error(
                        f'Failed to upload {oss_object_name} due to exceeded retries.'
                    )
                    raise e

        return object_key
