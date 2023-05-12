# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import unittest

from modelscope.hub.api import HubApi
from modelscope.msdatasets import MsDataset
from modelscope.msdatasets.dataset_cls.dataset import VirgoDataset
from modelscope.utils.constant import DownloadMode, Hubs, VirgoDatasetConfig
from modelscope.utils.logger import get_logger

logger = get_logger()

# Please use your own access token for buc account.
YOUR_ACCESS_TOKEN = 'your_access_token'
# Please use your own virgo dataset id and ensure you have access to it.
VIRGO_DATASET_ID = 'your_virgo_dataset_id'


class TestVirgoDataset(unittest.TestCase):

    def setUp(self):
        self.api = HubApi()
        self.api.login(YOUR_ACCESS_TOKEN)

    @unittest.skip('to be used for local test only')
    def test_download_virgo_dataset_meta(self):
        ds = MsDataset.load(dataset_name=VIRGO_DATASET_ID, hub=Hubs.virgo)
        ds_one = next(iter(ds))
        logger.info(ds_one)

        self.assertTrue(ds_one)
        self.assertIsInstance(ds, VirgoDataset)
        self.assertIn(VirgoDatasetConfig.col_id, ds_one)
        self.assertIn(VirgoDatasetConfig.col_meta_info, ds_one)
        self.assertIn(VirgoDatasetConfig.col_analysis_result, ds_one)
        self.assertIn(VirgoDatasetConfig.col_external_info, ds_one)

    @unittest.skip('to be used for local test only')
    def test_download_virgo_dataset_files(self):
        ds = MsDataset.load(
            dataset_name=VIRGO_DATASET_ID,
            hub=Hubs.virgo,
            download_virgo_files=True)

        ds_one = next(iter(ds))
        logger.info(ds_one)

        self.assertTrue(ds_one)
        self.assertIsInstance(ds, VirgoDataset)
        self.assertTrue(ds.download_virgo_files)
        self.assertIn(VirgoDatasetConfig.col_cache_file, ds_one)
        cache_file_path = ds_one[VirgoDatasetConfig.col_cache_file]
        self.assertTrue(os.path.exists(cache_file_path))

    @unittest.skip('to be used for local test only')
    def test_force_download_virgo_dataset_files(self):
        ds = MsDataset.load(
            dataset_name=VIRGO_DATASET_ID,
            hub=Hubs.virgo,
            download_mode=DownloadMode.FORCE_REDOWNLOAD,
            download_virgo_files=True)

        ds_one = next(iter(ds))
        logger.info(ds_one)

        self.assertTrue(ds_one)
        self.assertIsInstance(ds, VirgoDataset)
        self.assertTrue(ds.download_virgo_files)
        self.assertIn(VirgoDatasetConfig.col_cache_file, ds_one)
        cache_file_path = ds_one[VirgoDatasetConfig.col_cache_file]
        self.assertTrue(os.path.exists(cache_file_path))

    @unittest.skip('to be used for local test only')
    def test_download_virgo_dataset_odps(self):
        # Note: the samplingType must be 1, which means to get the dataset from MaxCompute(ODPS).
        import pandas as pd

        ds = MsDataset.load(
            dataset_name=VIRGO_DATASET_ID,
            hub=Hubs.virgo,
            odps_batch_size=100,
            odps_limit=2000,
            odps_drop_last=True)

        ds_one = next(iter(ds))
        logger.info(ds_one)

        self.assertTrue(ds_one)
        self.assertIsInstance(ds, VirgoDataset)
        self.assertTrue(ds_one, pd.DataFrame)
        logger.info(f'The shape of sample: {ds_one.shape}')


if __name__ == '__main__':
    unittest.main()
