# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.models.cv.image_inpainting import FFTInpainting
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.constant import ModelFile
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


class ImageInpaintingTrainerTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self.model_id = 'damo/cv_fft_inpainting_lama'
        self.cache_path = snapshot_download(self.model_id)
        cfg = Config.from_file(
            os.path.join(self.cache_path, ModelFile.CONFIGURATION))

        train_data_cfg = ConfigDict(
            name='PlacesToydataset',
            split='train',
            mask_gen_kwargs=cfg.dataset.mask_gen_kwargs,
            out_size=cfg.dataset.train_out_size,
            test_mode=False)

        test_data_cfg = ConfigDict(
            name='PlacesToydataset',
            split='test',
            mask_gen_kwargs=cfg.dataset.mask_gen_kwargs,
            out_size=cfg.dataset.val_out_size,
            test_mode=True)

        self.train_dataset = MsDataset.load(
            dataset_name=train_data_cfg.name,
            split=train_data_cfg.split,
            mask_gen_kwargs=train_data_cfg.mask_gen_kwargs,
            out_size=train_data_cfg.out_size,
            test_mode=train_data_cfg.test_mode)
        assert next(
            iter(self.train_dataset.config_kwargs['split_config'].values()))

        self.test_dataset = MsDataset.load(
            dataset_name=test_data_cfg.name,
            split=test_data_cfg.split,
            mask_gen_kwargs=test_data_cfg.mask_gen_kwargs,
            out_size=test_data_cfg.out_size,
            test_mode=test_data_cfg.test_mode)
        assert next(
            iter(self.test_dataset.config_kwargs['split_config'].values()))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_trainer(self):
        kwargs = dict(
            model=self.model_id,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset)

        trainer = build_trainer(
            name=Trainers.image_inpainting, default_args=kwargs)
        trainer.train()
        results_files = os.listdir(trainer.work_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)


if __name__ == '__main__':
    unittest.main()
