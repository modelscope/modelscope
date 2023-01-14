# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import os
import shutil
import tempfile
import unittest

import cv2
import json
import numpy as np
import torch

from modelscope.trainers.training_args import (ArgAttr, CliArgumentParser,
                                               training_args)
from modelscope.utils.test_utils import test_level


class TrainingArgsTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def tearDown(self):
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_define_args(self):
        myparser = CliArgumentParser(training_args)
        input_args = [
            '--max_epochs', '100', '--work_dir', 'ddddd', '--train_batch_size',
            '8', '--unkown', 'unkown'
        ]
        args, remainning = myparser.parse_known_args(input_args)
        myparser.print_help()
        self.assertTrue(args.max_epochs == 100)
        self.assertTrue(args.work_dir == 'ddddd')
        self.assertTrue(args.train_batch_size == 8)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_new_args(self):
        training_args.num_classes = ArgAttr(
            'model.mm_model.head.num_classes',
            type=int,
            help='number of classes')
        training_args.mean = ArgAttr(
            'train.data.mean', help='3-dim mean vector')
        training_args.flip = ArgAttr('train.data.flip', help='flip or not')
        training_args.img_size = ArgAttr(
            'train.data.img_size', help='image size')
        myparser = CliArgumentParser(training_args)
        input_args = [
            '--max_epochs', '100', '--work_dir', 'ddddd', '--train_batch_size',
            '8', '--num_classes', '10', '--mean', '[125.0,125.0,125.0]',
            '--flip', 'false', '--img_size', '(640,640)'
        ]
        args, remainning = myparser.parse_known_args(input_args)
        myparser.print_help()
        self.assertTrue(args.max_epochs == 100)
        self.assertTrue(args.work_dir == 'ddddd')
        self.assertTrue(args.train_batch_size == 8)
        self.assertTrue(args.num_classes == 10)
        self.assertTrue(len(args.mean) == 3)
        self.assertTrue(not args.flip)
        self.assertAlmostEqual(args.mean[0], 125.0)
        self.assertAlmostEqual(args.img_size, (640, 640))

        cfg_dict = myparser.get_cfg_dict(args=input_args)
        self.assertTrue(cfg_dict['model.mm_model.head.num_classes'] == 10)
        self.assertAlmostEqual(cfg_dict['train.data.mean'],
                               [125.0, 125.0, 125.0])
        self.assertTrue(not cfg_dict['train.data.flip'])
        self.assertEqual(cfg_dict['train.dataloader.batch_size_per_gpu'], 8)
        self.assertEqual(cfg_dict['train.work_dir'], 'ddddd')
        self.assertEqual(cfg_dict['train.max_epochs'], 100)
        self.assertEqual(cfg_dict['train.data.img_size'], (640, 640))


if __name__ == '__main__':
    unittest.main()
