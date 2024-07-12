# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import tempfile
import time
import unittest

from modelscope.hub.file_download import dataset_file_download
from modelscope.hub.snapshot_download import dataset_snapshot_download


class DownloadDatasetTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_dataset_file_download(self):
        dataset_id = 'citest/test_dataset_download'
        file_path = 'open_qa.jsonl'
        deep_file_path = '111/222/333/shijian.jpeg'
        start_time = time.time()

        #  test download to cache dir.
        with tempfile.TemporaryDirectory() as temp_cache_dir:
            #  first download to cache.
            cache_file_path = dataset_file_download(
                dataset_id=dataset_id,
                file_path=file_path,
                cache_dir=temp_cache_dir)
            file_modify_time = os.path.getmtime(cache_file_path)
            print(cache_file_path)
            assert cache_file_path == os.path.join(temp_cache_dir, dataset_id,
                                                   file_path)
            assert file_modify_time > start_time
            # download again, will get cached file.
            cache_file_path = dataset_file_download(
                dataset_id=dataset_id,
                file_path=file_path,
                cache_dir=temp_cache_dir)
            file_modify_time2 = os.path.getmtime(cache_file_path)
            assert file_modify_time == file_modify_time2

            deep_cache_file_path = dataset_file_download(
                dataset_id=dataset_id,
                file_path=deep_file_path,
                cache_dir=temp_cache_dir)
            deep_file_cath_path = os.path.join(temp_cache_dir, dataset_id,
                                               deep_file_path)
            assert deep_cache_file_path == deep_file_cath_path
            os.path.exists(deep_cache_file_path)

        # test download to local dir
        with tempfile.TemporaryDirectory() as temp_local_dir:
            #  first download to cache.
            cache_file_path = dataset_file_download(
                dataset_id=dataset_id,
                file_path=file_path,
                local_dir=temp_local_dir)
            assert cache_file_path == os.path.join(temp_local_dir, file_path)
            file_modify_time = os.path.getmtime(cache_file_path)
            assert file_modify_time > start_time
            # download again, will get cached file.
            cache_file_path = dataset_file_download(
                dataset_id=dataset_id,
                file_path=file_path,
                local_dir=temp_local_dir)
            file_modify_time2 = os.path.getmtime(cache_file_path)
            assert file_modify_time == file_modify_time2

    def test_dataset_snapshot_download(self):
        dataset_id = 'citest/test_dataset_download'
        file_path = 'open_qa.jsonl'
        deep_file_path = '111/222/333/shijian.jpeg'
        start_time = time.time()

        #  test download to cache dir.
        with tempfile.TemporaryDirectory() as temp_cache_dir:
            #  first download to cache.
            dataset_cache_path = dataset_snapshot_download(
                dataset_id=dataset_id, cache_dir=temp_cache_dir)
            file_modify_time = os.path.getmtime(
                os.path.join(dataset_cache_path, file_path))
            assert dataset_cache_path == os.path.join(temp_cache_dir,
                                                      dataset_id)
            assert file_modify_time > start_time
            assert os.path.exists(
                os.path.join(temp_cache_dir, dataset_id, deep_file_path))

            # download again, will get cached file.
            dataset_cache_path2 = dataset_snapshot_download(
                dataset_id=dataset_id, cache_dir=temp_cache_dir)
            file_modify_time2 = os.path.getmtime(
                os.path.join(dataset_cache_path2, file_path))
            assert file_modify_time == file_modify_time2

        # test download to local dir
        with tempfile.TemporaryDirectory() as temp_local_dir:
            #  first download to cache.
            dataset_cache_path = dataset_snapshot_download(
                dataset_id=dataset_id, local_dir=temp_local_dir)
            # root path is temp_local_dir, file download to local_dir
            assert dataset_cache_path == temp_local_dir
            file_modify_time = os.path.getmtime(
                os.path.join(dataset_cache_path, file_path))
            assert file_modify_time > start_time
            # download again, will get cached file.
            dataset_cache_path2 = dataset_snapshot_download(
                dataset_id=dataset_id, local_dir=temp_local_dir)
            file_modify_time2 = os.path.getmtime(
                os.path.join(dataset_cache_path2, file_path))
            assert file_modify_time == file_modify_time2

        #  test download with wild pattern, ignore_file_pattern
        with tempfile.TemporaryDirectory() as temp_cache_dir:
            #  first download to cache.
            dataset_cache_path = dataset_snapshot_download(
                dataset_id=dataset_id,
                cache_dir=temp_cache_dir,
                ignore_file_pattern='*.jpeg')
            assert dataset_cache_path == os.path.join(temp_cache_dir,
                                                      dataset_id)
            assert not os.path.exists(
                os.path.join(temp_cache_dir, dataset_id, deep_file_path))
            assert not os.path.exists(
                os.path.join(temp_cache_dir, dataset_id, '111/shijian.jpeg'))
            assert not os.path.exists(
                os.path.join(temp_cache_dir, dataset_id,
                             '111/222/shijian.jpeg'))
            assert os.path.exists(
                os.path.join(temp_cache_dir, dataset_id, file_path))

        #  test download with wild pattern, allow_file_pattern
        with tempfile.TemporaryDirectory() as temp_cache_dir:
            #  first download to cache.
            dataset_cache_path = dataset_snapshot_download(
                dataset_id=dataset_id,
                cache_dir=temp_cache_dir,
                allow_file_pattern='*.jpeg')
            assert dataset_cache_path == os.path.join(temp_cache_dir,
                                                      dataset_id)
            assert os.path.exists(
                os.path.join(temp_cache_dir, dataset_id, deep_file_path))
            assert os.path.exists(
                os.path.join(temp_cache_dir, dataset_id, '111/shijian.jpeg'))
            assert os.path.exists(
                os.path.join(temp_cache_dir, dataset_id,
                             '111/222/shijian.jpeg'))
            assert not os.path.exists(
                os.path.join(temp_cache_dir, dataset_id, file_path))

        # test download with wild pattern, allow_file_pattern and ignore file pattern.
        with tempfile.TemporaryDirectory() as temp_cache_dir:
            #  first download to cache.
            dataset_cache_path = dataset_snapshot_download(
                dataset_id=dataset_id,
                cache_dir=temp_cache_dir,
                ignore_file_pattern='*.jpeg',
                allow_file_pattern='*.xxx')
            assert dataset_cache_path == os.path.join(temp_cache_dir,
                                                      dataset_id)
            assert not os.path.exists(
                os.path.join(temp_cache_dir, dataset_id, deep_file_path))
            assert not os.path.exists(
                os.path.join(temp_cache_dir, dataset_id, '111/shijian.jpeg'))
            assert not os.path.exists(
                os.path.join(temp_cache_dir, dataset_id,
                             '111/222/shijian.jpeg'))
            assert not os.path.exists(
                os.path.join(temp_cache_dir, dataset_id, file_path))
