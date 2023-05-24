# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import json

from modelscope import MsDataset, TrainingArgs, build_dataset_from_file
from modelscope.utils.test_utils import test_level


class TestCli(unittest.TestCase):

    def setUp(self) -> None:
        content = [{
            'dataset': {
                'dataset_name': 'clue',
                'subset_name': 'cmnli',
                'split': 'train',
            },
            'column_mapping': {
                'sentence1': 'sentence1',
                'sentence2': 'sentence2',
                'label': 'label',
            },
            'split': 0.8,
        }, {
            'dataset': {
                'dataset_name': 'glue',
                'subset_name': 'mnli',
                'split': 'validation_matched',
            },
            'column_mapping': {
                'premise': 'sentence1',
                'hypothesis': 'sentence2',
                'label': 'label',
            },
            'split': 'val',
        }]
        with open('./dataset.json', 'w') as f:
            json.dump(content, f)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_merge_dataset_from_file(self):
        dataset = MsDataset.load('clue', subset_name='cmnli', split='train')
        dataset2 = MsDataset.load(
            'glue', subset_name='mnli', split='validation_matched')
        training_args = TrainingArgs(dataset_json_file='./dataset.json')
        train, test = build_dataset_from_file(training_args.dataset_json_file)
        self.assertEqual(len(train) + len(test), len(dataset) + len(dataset2))


if __name__ == '__main__':
    unittest.main()
