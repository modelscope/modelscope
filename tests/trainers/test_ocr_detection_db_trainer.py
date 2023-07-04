# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import os
import shutil
import tempfile
import unittest

import torch

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.pipelines import pipeline
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.test_utils import DistributedTestCase, test_level


def _setup():
    model_id = 'damo/cv_resnet18_ocr-detection-db-line-level_damo'
    cache_path = snapshot_download(model_id)
    return cache_path


class TestOCRDetectionDBTrainerSingleGPU(unittest.TestCase):

    def setUp(self):
        self.model_id = 'damo/cv_resnet18_ocr-detection-db-line-level_damo'
        self.test_image = 'data/test/images/ocr_detection/test_images/X51007339105.jpg'
        self.cache_path = _setup()
        self.config_file = os.path.join(self.cache_path, 'configuration.json')
        self.pretrained_model = os.path.join(
            self.cache_path, 'db_resnet18_public_line_640x640.pt')
        self.saved_dir = './workdirs'
        self.saved_finetune_model = os.path.join(self.saved_dir, 'final.pt')
        self.saved_infer_model = os.path.join(self.saved_dir,
                                              'pytorch_model.pt')

    def tearDown(self):
        shutil.rmtree(self.saved_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer_finetune_singleGPU(self):

        kwargs = dict(
            cfg_file=self.config_file,
            gpu_ids=[
                0,
            ],
            batch_size=8,
            max_epochs=5,
            base_lr=0.007,
            load_pretrain=True,
            pretrain_model=self.pretrained_model,
            cache_path=self.cache_path,
            train_data_dir=['./data/test/images/ocr_detection/'],
            train_data_list=[
                './data/test/images/ocr_detection/train_list.txt'
            ],
            val_data_dir=['./data/test/images/ocr_detection/'],
            val_data_list=['./data/test/images/ocr_detection/test_list.txt'])
        trainer = build_trainer(
            name=Trainers.ocr_detection_db, default_args=kwargs)
        trainer.train()
        trainer.evaluate(checkpoint_path=self.saved_finetune_model)

        # inference with pipeline using saved inference model
        cmd = 'cp {} {}'.format(self.config_file, self.saved_dir)
        os.system(cmd)
        ocr_detection = pipeline(Tasks.ocr_detection, model=self.saved_dir)
        result = ocr_detection(self.test_image)
        print('ocr detection results: ')
        print(result)


if __name__ == '__main__':
    unittest.main()
