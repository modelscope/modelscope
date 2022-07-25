# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
import shutil
import tempfile
import unittest
from typing import Callable, List, Optional, Tuple, Union

import cv2
import torch
from torch.utils import data as data

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models.cv.image_color_enhance.image_color_enhance import \
    ImageColorEnhance
from modelscope.trainers import build_trainer
from modelscope.utils.constant import ModelFile
from modelscope.utils.test_utils import test_level


class TestImageColorEnhanceTrainer(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self.model_id = 'damo/cv_csrnet_image-color-enhance-models'

        class PairedImageDataset(data.Dataset):

            def __init__(self, root):
                super(PairedImageDataset, self).__init__()
                gt_dir = osp.join(root, 'gt')
                lq_dir = osp.join(root, 'lq')
                self.gt_filelist = os.listdir(gt_dir)
                self.gt_filelist = sorted(
                    self.gt_filelist, key=lambda x: int(x[:-4]))
                self.gt_filelist = [
                    osp.join(gt_dir, f) for f in self.gt_filelist
                ]
                self.lq_filelist = os.listdir(lq_dir)
                self.lq_filelist = sorted(
                    self.lq_filelist, key=lambda x: int(x[:-4]))
                self.lq_filelist = [
                    osp.join(lq_dir, f) for f in self.lq_filelist
                ]

            def _img_to_tensor(self, img):
                return torch.from_numpy(img[:, :, [2, 1, 0]]).permute(
                    2, 0, 1).type(torch.float32) / 255.

            def __getitem__(self, index):
                lq = cv2.imread(self.lq_filelist[index])
                gt = cv2.imread(self.gt_filelist[index])
                lq = cv2.resize(lq, (256, 256), interpolation=cv2.INTER_CUBIC)
                gt = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_CUBIC)
                return \
                    {'src': self._img_to_tensor(lq), 'target': self._img_to_tensor(gt)}

            def __len__(self):
                return len(self.gt_filelist)

            def to_torch_dataset(self,
                                 columns: Union[str, List[str]] = None,
                                 preprocessors: Union[Callable,
                                                      List[Callable]] = None,
                                 **format_kwargs):
                return self

        self.dataset = PairedImageDataset(
            './data/test/images/image_color_enhance/')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer(self):
        kwargs = dict(
            model=self.model_id,
            train_dataset=self.dataset,
            eval_dataset=self.dataset,
            work_dir=self.tmp_dir)

        trainer = build_trainer(default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(3):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_trainer_with_model_and_args(self):
        cache_path = snapshot_download(self.model_id)
        model = ImageColorEnhance.from_pretrained(cache_path)
        kwargs = dict(
            cfg_file=os.path.join(cache_path, ModelFile.CONFIGURATION),
            model=model,
            train_dataset=self.dataset,
            eval_dataset=self.dataset,
            max_epochs=2,
            work_dir=self.tmp_dir)

        trainer = build_trainer(default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(2):
            self.assertIn(f'epoch_{i+1}.pth', results_files)


if __name__ == '__main__':
    unittest.main()
