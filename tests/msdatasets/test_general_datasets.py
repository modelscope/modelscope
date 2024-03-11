# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope import MsDataset
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import require_tf, require_torch, test_level

logger = get_logger()

# Note: MODELSCOPE_DOMAIN is set to 'test.modelscope.cn' in the environment variable

TEST_INNER_LEVEL = 1


class GeneralMsDatasetTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= TEST_INNER_LEVEL,
                         'skip test in current test level')
    def test_return_dataset_info_only(self):
        ds = MsDataset.load(
            'wangxingjun778test/aya_dataset_mini', dataset_info_only=True)
        print(f'>>output of test_return_dataset_info_only:\n {ds}')

    @unittest.skipUnless(test_level() >= TEST_INNER_LEVEL,
                         'skip test in current test level')
    def test_inner_fashion_mnist(self):
        # inner means the dataset is on the test.modelscope.cn environment
        ds = MsDataset.load(
            'xxxxtest0004/ms_test_0308_py',
            subset_name='fashion_mnist',
            split='train')
        print(f'>>output of test_inner_fashion_mnist:\n {next(iter(ds))}')

    @unittest.skipUnless(test_level() >= TEST_INNER_LEVEL,
                         'skip test in current test level')
    def test_inner_clue(self):
        ds = MsDataset.load(
            'wangxingjun778test/clue', subset_name='afqmc', split='train')
        print(f'>>output of test_inner_clue:\n {next(iter(ds))}')

    @unittest.skipUnless(test_level() >= TEST_INNER_LEVEL,
                         'skip test in current test level')
    def test_inner_cats_and_dogs_mini(self):
        ds = MsDataset.load(
            'wangxingjun778test/cats_and_dogs_mini', split='train')
        print(f'>>output of test_inner_cats_and_dogs_mini:\n {next(iter(ds))}')

    @unittest.skipUnless(test_level() >= TEST_INNER_LEVEL,
                         'skip test in current test level')
    def test_inner_aya_dataset_mini(self):
        ds = MsDataset.load(
            'wangxingjun778test/aya_dataset_mini', split='train')
        print(f'>>output of test_inner_aya_dataset_mini:\n {next(iter(ds))}')

        ds = MsDataset.load(
            'wangxingjun778test/aya_dataset_mini', subset_name='demographics')
        assert next(iter(ds['train']))
        print(
            f">>output of test_inner_aya_dataset_mini:\n {next(iter(ds['train']))}"
        )


if __name__ == '__main__':
    unittest.main()
