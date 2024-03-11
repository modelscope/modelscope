# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope import MsDataset
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import require_tf, require_torch, test_level

logger = get_logger()

# Note: MODELSCOPE_DOMAIN is set to 'test.modelscope.cn' in the environment variable


class GeneralMsDatasetTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_inner_fashion_mnist(self):
        # inner means the dataset is on the test.modelscope.cn environment
        ds = MsDataset.load(
            'xxxxtest0004/ms_test_0308_py',
            subset_name='fashion_mnist',
            split='train')
        logger.info(f'>>output:\n {next(iter(ds))}')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_inner_clue(self):
        ds = MsDataset.load(
            'wangxingjun778test/clue', subset_name='afqmc', split='train')
        logger.info(f'>>output:\n {next(iter(ds))}')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_inner_cats_and_dogs_mini(self):
        ds = MsDataset.load(
            'wangxingjun778test/cats_and_dogs_mini', split='train')
        logger.info(f'>>output:\n {next(iter(ds))}')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_inner_aya_dataset_mini(self):
        # TODO: subset_name='demographics'
        ds = MsDataset.load(
            'wangxingjun778test/aya_dataset_mini', split='train')
        logger.info(f'>>output:\n {next(iter(ds))}')


if __name__ == '__main__':
    unittest.main()
