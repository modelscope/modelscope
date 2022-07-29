from __future__ import print_function
import os
import sys

import oss2
from datasets.utils.file_utils import hash_url_to_filename


class OssUtilities:

    def __init__(self, oss_config):
        self.key = oss_config['AccessId']
        self.secret = oss_config['AccessSecret']
        self.token = oss_config['SecurityToken']
        self.endpoint = f"https://{oss_config['Region']}.aliyuncs.com"
        self.bucket_name = oss_config['Bucket']
        auth = oss2.StsAuth(self.key, self.secret, self.token)
        self.bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)
        self.oss_dir = oss_config['Dir']
        self.oss_backup_dir = oss_config['BackupDir']

    def download(self, oss_file_name, cache_dir):
        candidate_key = os.path.join(self.oss_dir, oss_file_name)
        candidate_key_backup = os.path.join(self.oss_backup_dir, oss_file_name)
        file_oss_key = candidate_key if self.bucket.object_exists(
            candidate_key) else candidate_key_backup
        filename = hash_url_to_filename(file_oss_key, etag=None)
        local_path = os.path.join(cache_dir, filename)

        def percentage(consumed_bytes, total_bytes):
            if total_bytes:
                rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
                print('\r{0}% '.format(rate), end='', flush=True)

        self.bucket.get_object_to_file(
            file_oss_key, local_path, progress_callback=percentage)
        return local_path
