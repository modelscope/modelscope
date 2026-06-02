# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models.cv.video_summarization import PGLVideoSummarization
from modelscope.msdatasets.dataset_cls.custom_datasets import \
    VideoSummarizationDataset
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile
from modelscope.utils.import_utils import exists
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


class VideoSummarizationTrainerTest(unittest.TestCase):
    # TODO: To be added to CUSTOM_DATASETS register

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self.model_id = 'damo/cv_googlenet_pgl-video-summarization'
        self.cache_path = snapshot_download(self.model_id)
        self.config = Config.from_file(
            os.path.join(self.cache_path, ModelFile.CONFIGURATION))
        self.dataset_train = VideoSummarizationDataset('train',
                                                       self.config.dataset,
                                                       self.cache_path)
        self.dataset_val = VideoSummarizationDataset('test',
                                                     self.config.dataset,
                                                     self.cache_path)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    @unittest.skipUnless(
        exists('transformers<5.0'),
        'Skip test because transformers version is too high.')
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

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_trainer_with_model_and_args(self):
        model = PGLVideoSummarization.from_pretrained(self.cache_path)
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
