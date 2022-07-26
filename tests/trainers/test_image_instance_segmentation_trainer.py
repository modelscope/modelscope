# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest
import zipfile
from functools import partial

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models.cv.image_instance_segmentation import \
    CascadeMaskRCNNSwinModel
from modelscope.models.cv.image_instance_segmentation.datasets import \
    ImageInstanceSegmentationCocoDataset
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile
from modelscope.utils.test_utils import test_level


class TestImageInstanceSegmentationTrainer(unittest.TestCase):

    model_id = 'damo/cv_swin-b_image-instance-segmentation_coco'

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

        cache_path = snapshot_download(self.model_id)
        config_path = os.path.join(cache_path, ModelFile.CONFIGURATION)
        cfg = Config.from_file(config_path)

        data_root = cfg.dataset.data_root
        classes = tuple(cfg.dataset.classes)
        max_epochs = cfg.train.max_epochs
        samples_per_gpu = cfg.train.dataloader.batch_size_per_gpu

        if data_root is None:
            # use default toy data
            dataset_path = os.path.join(cache_path, 'toydata.zip')
            with zipfile.ZipFile(dataset_path, 'r') as zipf:
                zipf.extractall(cache_path)
            data_root = cache_path + '/toydata/'
            classes = ('Cat', 'Dog')

        self.train_dataset = ImageInstanceSegmentationCocoDataset(
            data_root + 'annotations/instances_train.json',
            classes=classes,
            data_root=data_root,
            img_prefix=data_root + 'images/train/',
            seg_prefix=None,
            test_mode=False)

        self.eval_dataset = ImageInstanceSegmentationCocoDataset(
            data_root + 'annotations/instances_val.json',
            classes=classes,
            data_root=data_root,
            img_prefix=data_root + 'images/val/',
            seg_prefix=None,
            test_mode=True)

        from mmcv.parallel import collate

        self.collate_fn = partial(collate, samples_per_gpu=samples_per_gpu)

        self.max_epochs = max_epochs

        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_trainer(self):
        kwargs = dict(
            model=self.model_id,
            data_collator=self.collate_fn,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            work_dir=self.tmp_dir)

        trainer = build_trainer(
            name='image-instance-segmentation', default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_trainer_with_model_and_args(self):
        tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        cache_path = snapshot_download(self.model_id)
        model = CascadeMaskRCNNSwinModel.from_pretrained(cache_path)
        kwargs = dict(
            cfg_file=os.path.join(cache_path, ModelFile.CONFIGURATION),
            model=model,
            data_collator=self.collate_fn,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            work_dir=self.tmp_dir)

        trainer = build_trainer(
            name='image-instance-segmentation', default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)


if __name__ == '__main__':
    unittest.main()
