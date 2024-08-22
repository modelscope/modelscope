# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope import MsDataset
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


class TestStreamLoad(unittest.TestCase):

    def setUp(self):
        ...

    def tearDown(self):
        ...

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_stream_read_zstd(self):
        repo_id: str = 'swift/chinese-c4'
        ds = MsDataset.load(repo_id, split='train', use_streaming=True)
        sample = next(iter(ds))
        logger.info(sample)

        assert sample['url'], f'Failed to load sample from {repo_id}'

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_stream_imagefolder(self):
        repo_id: str = 'wangxingjun778/test_new_dataset'
        ds = MsDataset.load(repo_id, split='train', use_streaming=True)
        sample = next(iter(ds))
        logger.info(sample)

        assert sample['image'], f'Failed to load sample from {repo_id}'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_stream_parquet(self):
        repo_id: str = 'swift/A-OKVQA'
        ds = MsDataset.load(repo_id, split='train', use_streaming=True)
        sample = next(iter(ds))
        logger.info(sample)

        assert sample['question'], f'Failed to load sample from {repo_id}'


if __name__ == '__main__':
    unittest.main()
