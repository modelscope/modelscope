# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models.cv.image_deblur import NAFNetForImageDeblur
from modelscope.msdatasets import MsDataset
from modelscope.msdatasets.task_datasets.gopro_image_deblurring_dataset import \
    GoproImageDeblurringDataset
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config
from modelscope.utils.constant import DownloadMode, ModelFile
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


class ImageDeblurTrainerTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self.model_id = 'damo/cv_nafnet_image-deblur_gopro'
        self.cache_path = snapshot_download(self.model_id)
        self.config = Config.from_file(
            os.path.join(self.cache_path, ModelFile.CONFIGURATION))
        dataset_train = MsDataset.load(
            'GOPRO',
            namespace='damo',
            subset_name='default',
            split='test',
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)._hf_ds
        dataset_val = MsDataset.load(
            'GOPRO',
            namespace='damo',
            subset_name='subset',
            split='test',
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)._hf_ds
        self.dataset_train = GoproImageDeblurringDataset(
            dataset_train, self.config.dataset, is_train=True)
        self.dataset_val = GoproImageDeblurringDataset(
            dataset_val, self.config.dataset, is_train=False)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
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
        for i in range(1):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_trainer_with_model_and_args(self):
        model = NAFNetForImageDeblur.from_pretrained(self.cache_path)
        kwargs = dict(
            cfg_file=os.path.join(self.cache_path, ModelFile.CONFIGURATION),
            model=model,
            train_dataset=self.dataset_train,
            eval_dataset=self.dataset_val,
            max_epochs=1,
            work_dir=self.tmp_dir)
        trainer = build_trainer(default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(1):
            self.assertIn(f'epoch_{i+1}.pth', results_files)


if __name__ == '__main__':
    unittest.main()
