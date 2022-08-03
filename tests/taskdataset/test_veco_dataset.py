# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.task_datasets.veco_dataset import VecoDataset
from modelscope.utils.test_utils import test_level


class TestVecoDataset(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_veco_dataset_train(self):
        from datasets import Dataset
        d0 = Dataset.from_dict({'a': [0, 1, 2]})
        d1 = Dataset.from_dict({'a': [10, 11, 12, 13, 14]})
        d2 = Dataset.from_dict({'a': [21, 22, 23, 24, 25, 26, 27]})
        dataset = VecoDataset([d0, d1, d2], mode='train')
        self.assertEqual(len(dataset), 15)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_veco_dataset_eval(self):
        from datasets import Dataset
        d0 = Dataset.from_dict({'a': [0, 1, 2]})
        d1 = Dataset.from_dict({'a': [10, 11, 12, 13, 14]})
        d2 = Dataset.from_dict({'a': [21, 22, 23, 24, 25, 26, 27]})
        dataset = VecoDataset([d0, d1, d2], mode='eval')
        self.assertEqual(len(dataset), 3)
        dataset.switch_dataset(1)
        self.assertEqual(len(dataset), 5)
        dataset.switch_dataset(2)
        self.assertEqual(len(dataset), 7)


if __name__ == '__main__':
    unittest.main()
