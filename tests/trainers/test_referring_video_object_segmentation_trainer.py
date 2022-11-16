# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest
import zipfile

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.models.cv.referring_video_object_segmentation import \
    ReferringVideoObjectSegmentation
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.constant import ModelFile
from modelscope.utils.test_utils import test_level


class TestImageInstanceSegmentationTrainer(unittest.TestCase):

    model_id = 'damo/cv_swin-t_referring_video-object-segmentation'
    dataset_name = 'referring_vos_toydata'

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

        cache_path = snapshot_download(self.model_id)
        config_path = os.path.join(cache_path, ModelFile.CONFIGURATION)
        cfg = Config.from_file(config_path)

        max_epochs = cfg.train.max_epochs

        train_data_cfg = ConfigDict(
            name=self.dataset_name,
            split='train',
            test_mode=False,
            cfg=cfg.dataset)

        test_data_cfg = ConfigDict(
            name=self.dataset_name,
            split='test',
            test_mode=True,
            cfg=cfg.dataset)

        self.train_dataset = MsDataset.load(
            dataset_name=train_data_cfg.name,
            split=train_data_cfg.split,
            cfg=train_data_cfg.cfg,
            test_mode=train_data_cfg.test_mode)
        assert next(
            iter(self.train_dataset.config_kwargs['split_config'].values()))

        self.test_dataset = MsDataset.load(
            dataset_name=test_data_cfg.name,
            split=test_data_cfg.split,
            cfg=test_data_cfg.cfg,
            test_mode=test_data_cfg.test_mode)
        assert next(
            iter(self.test_dataset.config_kwargs['split_config'].values()))

        self.max_epochs = max_epochs

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer(self):
        kwargs = dict(
            model=self.model_id,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            work_dir='./work_dir')

        trainer = build_trainer(
            name=Trainers.referring_video_object_segmentation,
            default_args=kwargs)
        trainer.train()
        results_files = os.listdir(trainer.work_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_trainer_with_model_and_args(self):

        cache_path = snapshot_download(self.model_id)
        model = ReferringVideoObjectSegmentation.from_pretrained(cache_path)
        kwargs = dict(
            cfg_file=os.path.join(cache_path, ModelFile.CONFIGURATION),
            model=model,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            work_dir='./work_dir')

        trainer = build_trainer(
            name=Trainers.referring_video_object_segmentation,
            default_args=kwargs)
        trainer.train()
        results_files = os.listdir(trainer.work_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)


if __name__ == '__main__':
    unittest.main()
