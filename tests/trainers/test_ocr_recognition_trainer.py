# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.models.cv.ocr_recognition import OCRRecognition
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.constant import DownloadMode, ModelFile
from modelscope.utils.test_utils import test_level


@unittest.skip(
    "For FileNotFoundError: [Errno 2] No such file or directory: './work_dir/output/pytorch_model.pt' issue"
)
class TestOCRRecognitionTrainer(unittest.TestCase):

    model_id = 'damo/cv_crnn_ocr-recognition-general_damo'

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

        cache_path = snapshot_download(self.model_id, revision='v2.2.2')
        config_path = os.path.join(cache_path, ModelFile.CONFIGURATION)
        cfg = Config.from_file(config_path)

        max_epochs = cfg.train.max_epochs

        train_data_cfg = ConfigDict(
            name='ICDAR13_HCTR_Dataset', split='test', namespace='damo')

        test_data_cfg = ConfigDict(
            name='ICDAR13_HCTR_Dataset', split='test', namespace='damo')

        self.train_dataset = MsDataset.load(
            dataset_name=train_data_cfg.name,
            split=train_data_cfg.split,
            namespace=train_data_cfg.namespace,
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
        assert next(
            iter(self.train_dataset.config_kwargs['split_config'].values()))

        self.test_dataset = MsDataset.load(
            dataset_name=test_data_cfg.name,
            split=test_data_cfg.split,
            namespace=train_data_cfg.namespace,
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
        assert next(
            iter(self.test_dataset.config_kwargs['split_config'].values()))

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
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            work_dir=self.tmp_dir)

        trainer = build_trainer(
            name=Trainers.ocr_recognition, default_args=kwargs)
        trainer.train()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer_with_model_and_args(self):
        tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        cache_path = snapshot_download(self.model_id, revision='v2.2.2')
        model = OCRRecognition.from_pretrained(cache_path)
        kwargs = dict(
            cfg_file=os.path.join(cache_path, ModelFile.CONFIGURATION),
            model=model,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            work_dir=tmp_dir)

        trainer = build_trainer(
            name=Trainers.ocr_recognition, default_args=kwargs)
        trainer.train()


if __name__ == '__main__':
    unittest.main()
