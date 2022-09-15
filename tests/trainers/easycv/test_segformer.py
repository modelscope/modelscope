# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import os
import shutil
import tempfile
import unittest

import torch

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import LogKeys, Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level


@unittest.skipIf(not torch.cuda.is_available(), 'cuda unittest')
class EasyCVTrainerTestSegformer(unittest.TestCase):

    def setUp(self):
        self.logger = get_logger()
        self.logger.info(('Testing %s.%s' %
                          (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _train(self):
        # adapt to distributed mode
        from easycv.utils.test_util import pseudo_dist_init
        pseudo_dist_init()

        cfg_options = {'train.max_epochs': 2}

        trainer_name = Trainers.easycv
        train_dataset = MsDataset.load(
            dataset_name='small_coco_stuff164k',
            namespace='EasyCV',
            split='train')
        eval_dataset = MsDataset.load(
            dataset_name='small_coco_stuff164k',
            namespace='EasyCV',
            split='validation')
        kwargs = dict(
            model=
            'damo/cv_segformer-b0_image_semantic-segmentation_coco-stuff164k',
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            work_dir=self.tmp_dir,
            cfg_options=cfg_options)

        trainer = build_trainer(trainer_name, kwargs)
        trainer.train()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_single_gpu_segformer(self):
        self._train()

        results_files = os.listdir(self.tmp_dir)
        json_files = glob.glob(os.path.join(self.tmp_dir, '*.log.json'))
        self.assertEqual(len(json_files), 1)
        self.assertIn(f'{LogKeys.EPOCH}_1.pth', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_2.pth', results_files)


if __name__ == '__main__':
    unittest.main()
