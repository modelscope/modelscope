# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

from modelscope import MsDataset
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()

TEST_INNER_LEVEL = 1


class GeneralMsDatasetTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= TEST_INNER_LEVEL,
                         'skip test in current test level')
    def test_return_dataset_info_only(self):
        ds = MsDataset.load(
            'wangxingjun778/aya_dataset_mini', dataset_info_only=True)
        logger.info(f'>>output of test_return_dataset_info_only:\n {ds}')

    @unittest.skipUnless(test_level() >= TEST_INNER_LEVEL,
                         'skip test in current test level')
    def test_inner_fashion_mnist(self):
        # inner means the dataset is on the test.modelscope.cn environment
        ds = MsDataset.load(
            'wangxingjun778/ms_test_0308_py',
            subset_name='fashion_mnist',
            split='train')
        logger.info(
            f'>>output of test_inner_fashion_mnist:\n {next(iter(ds))}')

    @unittest.skipUnless(test_level() >= TEST_INNER_LEVEL,
                         'skip test in current test level')
    def test_inner_clue(self):
        ds = MsDataset.load(
            'wangxingjun778/clue', subset_name='afqmc', split='train')
        logger.info(f'>>output of test_inner_clue:\n {next(iter(ds))}')

    @unittest.skipUnless(test_level() >= TEST_INNER_LEVEL,
                         'skip test in current test level')
    def test_inner_cats_and_dogs_mini(self):
        ds = MsDataset.load('wangxingjun778/cats_and_dogs_mini', split='train')
        logger.info(
            f'>>output of test_inner_cats_and_dogs_mini:\n {next(iter(ds))}')

    @unittest.skipUnless(test_level() >= TEST_INNER_LEVEL,
                         'skip test in current test level')
    def test_inner_aya_dataset_mini(self):
        # Dataset Format:
        # data/train-xxx-of-xxx.parquet; data/test-xxx-of-xxx.parquet
        # demographics/train-xxx-of-xxx.parquet

        ds = MsDataset.load('wangxingjun778/aya_dataset_mini', split='train')
        logger.info(
            f'>>output of test_inner_aya_dataset_mini:\n {next(iter(ds))}')

        ds = MsDataset.load(
            'wangxingjun778/aya_dataset_mini', subset_name='demographics')
        assert next(iter(ds['train']))
        logger.info(
            f">>output of test_inner_aya_dataset_mini:\n {next(iter(ds['train']))}"
        )

    @unittest.skipUnless(test_level() >= TEST_INNER_LEVEL,
                         'skip test in current test level')
    def test_inner_no_standard_imgs(self):
        infos = MsDataset.load(
            'wangxingjun778/png_jpg_txt_test', dataset_info_only=True)
        assert infos['default']

        ds = MsDataset.load('wangxingjun778/png_jpg_txt_test', split='train')
        logger.info(
            f'>>>output of test_inner_no_standard_imgs: \n{next(iter(ds))}')
        assert next(iter(ds))

    @unittest.skipUnless(test_level() >= 3, 'skip test in current test level')
    def test_inner_speech_yinpin(self):
        ds = MsDataset.load('wangxingjun778/hf_lj_speech_yinpin_test')
        logger.info(ds)
        assert next(iter(ds))

    @unittest.skipUnless(test_level() >= TEST_INNER_LEVEL,
                         'skip test in current test level')
    def test_inner_yuancheng_picture(self):
        ds = MsDataset.load(
            'wangxingjun778/yuancheng_picture',
            subset_name='remote_images',
            split='train')
        logger.info(next(iter(ds)))
        assert next(iter(ds))

    @unittest.skipUnless(test_level() >= TEST_INNER_LEVEL,
                         'skip test in current test level')
    def test_youku_mplug_dataset(self):
        # To test the Youku-AliceMind dataset with new sdk version
        ds = MsDataset.load(
            'modelscope/Youku-AliceMind',
            subset_name='classification',
            split='validation',  # Options: train, test, validation
            use_streaming=True)

        logger.info(next(iter(ds)))
        data_sample = next(iter(ds))

        assert data_sample['video_id'][0]
        assert os.path.exists(data_sample['video_id:FILE'][0])


if __name__ == '__main__':
    unittest.main()
