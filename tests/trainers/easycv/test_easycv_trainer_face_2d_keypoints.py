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
from modelscope.utils.constant import DownloadMode, LogKeys, Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level


@unittest.skipIf(not torch.cuda.is_available(), 'cuda unittest')
class EasyCVTrainerTestFace2DKeypoints(unittest.TestCase):
    model_id = 'damo/cv_mobilenet_face-2d-keypoints_alignment'

    def setUp(self):
        self.logger = get_logger()
        self.logger.info(('Testing %s.%s' %
                          (type(self).__name__, self._testMethodName)))

    def _train(self, tmp_dir):
        cfg_options = {'train.max_epochs': 2}

        trainer_name = Trainers.easycv

        train_dataset = MsDataset.load(
            dataset_name='face_2d_keypoints_dataset',
            namespace='modelscope',
            split='train',
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
        eval_dataset = MsDataset.load(
            dataset_name='face_2d_keypoints_dataset',
            namespace='modelscope',
            split='train',
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)

        kwargs = dict(
            model=self.model_id,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            work_dir=tmp_dir,
            cfg_options=cfg_options)

        trainer = build_trainer(trainer_name, kwargs)
        trainer.train()

    @unittest.skip(
        'skip since face_2d_keypoints_dataset is set to private for now')
    def test_trainer_single_gpu(self):
        temp_file_dir = tempfile.TemporaryDirectory()
        tmp_dir = temp_file_dir.name
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        self._train(tmp_dir)

        results_files = os.listdir(tmp_dir)
        json_files = glob.glob(os.path.join(tmp_dir, '*.log.json'))
        self.assertEqual(len(json_files), 1)
        self.assertIn(f'{LogKeys.EPOCH}_2.pth', results_files)

        temp_file_dir.cleanup()


if __name__ == '__main__':
    unittest.main()
