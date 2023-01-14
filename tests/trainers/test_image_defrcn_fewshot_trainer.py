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
        kwargs = dict(
            model=self.model_id,
            data_dir=self.data_dir,
            work_dir=self.tmp_dir,
            model_weights=os.path.join(get_cache_dir(), self.model_id,
                                       'ImageNetPretrained/MSRA/R-101.pkl'),
            data_type='pascal_voc',
            config_path='defrcn_det_r101_base{}.yaml'.format(split),
            datasets_train=('voc_2007_trainval_base{}'.format(split),
                            'voc_2012_trainval_base{}'.format(split)),
            datasets_test=('voc_2007_test_base{}'.format(split), ))
        trainer = build_trainer(
            name=Trainers.image_fewshot_detection, default_args=kwargs)
        trainer.train()

        results_files = os.listdir(self.tmp_dir)
        self.assertIn('metrics.json', results_files)
        self.assertIn('model_final.pth', results_files)


if __name__ == '__main__':
    unittest.main()
