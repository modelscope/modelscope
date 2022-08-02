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
from modelscope.models.cv.image_portrait_enhancement import \
    ImagePortraitEnhancement
from modelscope.trainers import build_trainer
from modelscope.utils.constant import ModelFile
from modelscope.utils.test_utils import test_level


class TestImagePortraitEnhancementTrainer(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self.model_id = 'damo/cv_gpen_image-portrait-enhancement'

        class PairedImageDataset(data.Dataset):

            def __init__(self, root, size=512):
                super(PairedImageDataset, self).__init__()
                self.size = size
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
                img = torch.from_numpy(img[:, :, [2, 1, 0]]).permute(
                    2, 0, 1).type(torch.float32) / 255.
                return (img - 0.5) / 0.5

            def __getitem__(self, index):
                lq = cv2.imread(self.lq_filelist[index])
                gt = cv2.imread(self.gt_filelist[index])
                lq = cv2.resize(
                    lq, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
                gt = cv2.resize(
                    gt, (self.size, self.size), interpolation=cv2.INTER_CUBIC)

                return \
                    {'src': self._img_to_tensor(lq), 'target': self._img_to_tensor(gt)}

            def __len__(self):
                return len(self.gt_filelist)

            def to_torch_dataset(self,
                                 columns: Union[str, List[str]] = None,
                                 preprocessors: Union[Callable,
                                                      List[Callable]] = None,
                                 **format_kwargs):
                # self.preprocessor = preprocessors
                return self

        self.dataset = PairedImageDataset(
            './data/test/images/face_enhancement/')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer(self):
        kwargs = dict(
            model=self.model_id,
            train_dataset=self.dataset,
            eval_dataset=self.dataset,
            device='gpu',
            work_dir=self.tmp_dir)

        trainer = build_trainer(name='gpen', default_args=kwargs)
        trainer.train()

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_trainer_with_model_and_args(self):
        tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        cache_path = snapshot_download(self.model_id)
        model = ImagePortraitEnhancement.from_pretrained(cache_path)
        kwargs = dict(
            cfg_file=os.path.join(cache_path, ModelFile.CONFIGURATION),
            model=model,
            train_dataset=self.dataset,
            eval_dataset=self.dataset,
            device='gpu',
            max_epochs=2,
            work_dir=self.tmp_dir)

        trainer = build_trainer(name='gpen', default_args=kwargs)
        trainer.train()


if __name__ == '__main__':
    unittest.main()
