# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import os
import shutil
import tempfile
import unittest

import requests
import torch

from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.utils.constant import LogKeys, Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level
from modelscope.utils.torch_utils import is_master


def _download_data(url, save_dir):
    r = requests.get(url, verify=True)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    zip_name = os.path.split(url)[-1]
    save_path = os.path.join(save_dir, zip_name)
    with open(save_path, 'wb') as f:
        f.write(r.content)

    unpack_dir = os.path.join(save_dir, os.path.splitext(zip_name)[0])
    shutil.unpack_archive(save_path, unpack_dir)


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
        from modelscope.trainers.easycv.trainer import EasyCVEpochBasedTrainer

        url = 'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/datasets/small_coco_stuff164k.zip'
        data_dir = os.path.join(self.tmp_dir, 'data')
        if is_master():
            _download_data(url, data_dir)

        # adapt to ditributed mode
        from easycv.utils.test_util import pseudo_dist_init
        pseudo_dist_init()

        root_path = os.path.join(data_dir, 'small_coco_stuff164k')
        cfg_options = {
            'train.max_epochs':
            2,
            'dataset.train.data_source.img_root':
            os.path.join(root_path, 'train2017'),
            'dataset.train.data_source.label_root':
            os.path.join(root_path, 'annotations/train2017'),
            'dataset.train.data_source.split':
            os.path.join(root_path, 'train.txt'),
            'dataset.val.data_source.img_root':
            os.path.join(root_path, 'val2017'),
            'dataset.val.data_source.label_root':
            os.path.join(root_path, 'annotations/val2017'),
            'dataset.val.data_source.split':
            os.path.join(root_path, 'val.txt'),
        }

        trainer_name = Trainers.easycv
        kwargs = dict(
            task=Tasks.image_segmentation,
            model='EasyCV/EasyCV-Segformer-b0',
            work_dir=self.tmp_dir,
            cfg_options=cfg_options)

        trainer = build_trainer(trainer_name, kwargs)
        trainer.train()

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_single_gpu_segformer(self):
        self._train()

        results_files = os.listdir(self.tmp_dir)
        json_files = glob.glob(os.path.join(self.tmp_dir, '*.log.json'))
        self.assertEqual(len(json_files), 1)
        self.assertIn(f'{LogKeys.EPOCH}_1.pth', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_2.pth', results_files)


if __name__ == '__main__':
    unittest.main()
