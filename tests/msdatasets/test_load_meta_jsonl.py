# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.msdatasets import MsDataset
from modelscope.utils import logger as logging
from modelscope.utils.constant import DownloadMode
from modelscope.utils.test_utils import test_level

logger = logging.get_logger()


class TestLoadMetaJsonl(unittest.TestCase):

    def setUp(self):
        self.dataset_id = 'modelscope/ms_ds_meta_jsonlines'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_load_jsonl_in_meta(self):
        ds = MsDataset.load(
            self.dataset_id,
            split='test',
            download_mode=DownloadMode.FORCE_REDOWNLOAD)
        ds_one = next(iter(ds))
        logger.info(next(iter(ds)))
        assert ds_one['text']


if __name__ == '__main__':
    unittest.main()
