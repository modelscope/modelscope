# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models.cv.image_colorization import DDColorForImageColorization
from modelscope.msdatasets import MsDataset
from modelscope.msdatasets.dataset_cls.custom_datasets.image_colorization import \
    ImageColorizationDataset
from modelscope.pipelines import pipeline
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config
from modelscope.utils.constant import DownloadMode, ModelFile, Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


class ImageColorizationTrainerTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self.model_id = 'damo/cv_ddcolor_image-colorization'
        self.cache_path = snapshot_download(self.model_id)
        self.config = Config.from_file(
            os.path.join(self.cache_path, ModelFile.CONFIGURATION))
        dataset_train = MsDataset.load(
            'imagenet-val5k-image',
            namespace='damo',
            subset_name='default',
            split='validation',
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)._hf_ds
        dataset_val = MsDataset.load(
            'imagenet-val5k-image',
            namespace='damo',
            subset_name='default',
            split='validation',
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)._hf_ds
        self.dataset_train = ImageColorizationDataset(
            dataset_train, self.config.dataset, is_train=True)
        self.dataset_val = ImageColorizationDataset(
            dataset_val, self.config.dataset, is_train=False)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
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
        pipeline_colorization = pipeline(
            task=Tasks.image_colorization, model=f'{self.tmp_dir}/output')
        pipeline_colorization('data/test/images/marilyn_monroe_4.jpg')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_trainer_with_model_and_args(self):
        model = DDColorForImageColorization.from_pretrained(self.cache_path)
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
        pipeline_colorization = pipeline(
            task=Tasks.image_colorization, model=f'{self.tmp_dir}/output')
        pipeline_colorization('data/test/images/marilyn_monroe_4.jpg')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_evaluation(self):
        kwargs = dict(
            model=self.model_id,
            train_dataset=self.dataset_train,
            eval_dataset=self.dataset_val,
            work_dir=self.tmp_dir)
        trainer = build_trainer(default_args=kwargs)
        results = trainer.evaluate()
        print(results)


if __name__ == '__main__':
    unittest.main()
