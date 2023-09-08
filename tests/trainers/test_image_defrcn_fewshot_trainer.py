# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from modelscope.hub.utils.utils import get_cache_dir
from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import DownloadMode
from modelscope.utils.test_utils import test_level


@unittest.skip(
    "For detection2 compatible  module 'PIL.Image' has no attribute 'LINEAR'")
class TestImageDefrcnFewShotTrainer(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        cmd = [
            sys.executable, '-m', 'pip', 'install', 'detectron2==0.3', '-f',
            'https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html'
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self.model_id = 'damo/cv_resnet101_detection_fewshot-defrcn'

        data_voc = MsDataset.load(
            dataset_name='VOC_fewshot',
            namespace='shimin2023',
            split='train',
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
        self.data_dir = os.path.join(
            data_voc.config_kwargs['split_config']['train'], 'data')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer(self):

        split = 1

        def base_cfg_modify_fn(cfg):
            cfg.train.work_dir = self.tmp_dir

            cfg.model.roi_heads.backward_scale = 0.75
            cfg.model.roi_heads.num_classes = 15
            cfg.model.roi_heads.freeze_feat = False
            cfg.model.roi_heads.cls_dropout = False
            cfg.model.weights = os.path.join(
                get_cache_dir(), self.model_id,
                'ImageNetPretrained/MSRA/R-101.pkl')

            cfg.datasets.root = self.data_dir
            cfg.datasets.type = 'pascal_voc'
            cfg.datasets.train = [
                'voc_2007_trainval_base{}'.format(split),
                'voc_2012_trainval_base{}'.format(split)
            ]
            cfg.datasets.test = ['voc_2007_test_base{}'.format(split)]
            cfg.input.min_size_test = 50
            cfg.train.dataloader.ims_per_batch = 4
            cfg.train.max_iter = 300
            cfg.train.optimizer.lr = 0.001
            cfg.train.lr_scheduler.warmup_iters = 100

            cfg.test.pcb_enable = False
            return cfg

        kwargs = dict(model=self.model_id, cfg_modify_fn=base_cfg_modify_fn)
        trainer = build_trainer(
            name=Trainers.image_fewshot_detection, default_args=kwargs)
        trainer.train()

        results_files = os.listdir(self.tmp_dir)
        self.assertIn('metrics.json', results_files)
        self.assertIn('model_final.pth', results_files)


if __name__ == '__main__':
    unittest.main()
