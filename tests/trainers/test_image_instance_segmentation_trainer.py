# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest
import zipfile
from functools import partial

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.models.cv.image_instance_segmentation import \
    CascadeMaskRCNNSwinModel
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.constant import DownloadMode, ModelFile
from modelscope.utils.test_utils import test_level


class TestImageInstanceSegmentationTrainer(unittest.TestCase):

    model_id = 'damo/cv_swin-b_image-instance-segmentation_coco'

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

        cache_path = snapshot_download(self.model_id)
        config_path = os.path.join(cache_path, ModelFile.CONFIGURATION)
        cfg = Config.from_file(config_path)

        max_epochs = cfg.train.max_epochs
        samples_per_gpu = cfg.train.dataloader.batch_size_per_gpu
        try:
            train_data_cfg = cfg.dataset.train
            val_data_cfg = cfg.dataset.val
        except Exception:
            train_data_cfg = None
            val_data_cfg = None
        if train_data_cfg is None:
            # use default toy data
            train_data_cfg = ConfigDict(
                name='pets_small', split='train', test_mode=False)
        if val_data_cfg is None:
            val_data_cfg = ConfigDict(
                name='pets_small', split='validation', test_mode=True)

        self.train_dataset = MsDataset.load(
            dataset_name=train_data_cfg.name,
            split=train_data_cfg.split,
            test_mode=train_data_cfg.test_mode,
            download_mode=DownloadMode.FORCE_REDOWNLOAD)
        assert self.train_dataset.config_kwargs['classes']
        assert next(
            iter(self.train_dataset.config_kwargs['split_config'].values()))

        self.eval_dataset = MsDataset.load(
            dataset_name=val_data_cfg.name,
            split=val_data_cfg.split,
            test_mode=val_data_cfg.test_mode,
            download_mode=DownloadMode.FORCE_REDOWNLOAD)
        assert self.eval_dataset.config_kwargs['classes']
        assert next(
            iter(self.eval_dataset.config_kwargs['split_config'].values()))

        from mmcv.parallel import collate

        self.collate_fn = partial(collate, samples_per_gpu=samples_per_gpu)

        self.max_epochs = max_epochs

        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer(self):
        kwargs = dict(
            model=self.model_id,
            data_collator=self.collate_fn,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            work_dir=self.tmp_dir)

        trainer = build_trainer(
            name=Trainers.image_instance_segmentation, default_args=kwargs)
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
            name=Trainers.image_instance_segmentation, default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)


if __name__ == '__main__':
    unittest.main()
