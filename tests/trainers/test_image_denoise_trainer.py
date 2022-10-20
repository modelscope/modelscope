# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models.cv.image_denoise import NAFNetForImageDenoise
from modelscope.msdatasets import MsDataset
from modelscope.msdatasets.task_datasets.sidd_image_denoising import \
    SiddImageDenoisingDataset
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config
from modelscope.utils.constant import DownloadMode, ModelFile
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


class ImageDenoiseTrainerTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self.model_id = 'damo/cv_nafnet_image-denoise_sidd'
        self.cache_path = snapshot_download(self.model_id)
        self.config = Config.from_file(
            os.path.join(self.cache_path, ModelFile.CONFIGURATION))
        dataset_train = MsDataset.load(
            'SIDD',
            namespace='huizheng',
            subset_name='default',
            split='validation',
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)._hf_ds
        dataset_val = MsDataset.load(
            'SIDD',
            namespace='huizheng',
            subset_name='default',
            split='test',
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)._hf_ds
        self.dataset_train = SiddImageDenoisingDataset(
            dataset_train, self.config.dataset, is_train=True)
        self.dataset_val = SiddImageDenoisingDataset(
            dataset_val, self.config.dataset, is_train=False)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_trainer(self):
        kwargs = dict(
            model=self.model_id,
            train_dataset=self.dataset_train,
            eval_dataset=self.dataset_val,
            work_dir=self.tmp_dir)
        trainer = build_trainer(default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(2):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_trainer_with_model_and_args(self):
        model = NAFNetForImageDenoise.from_pretrained(self.cache_path)
        kwargs = dict(
            cfg_file=os.path.join(self.cache_path, ModelFile.CONFIGURATION),
            model=model,
            train_dataset=self.dataset_train,
            eval_dataset=self.dataset_val,
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
