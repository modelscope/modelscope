# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import shutil
import tempfile
import unittest

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.audio.audio_utils import TtsTrainType
from modelscope.utils.constant import DownloadMode, Fields, Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


class TestASRTrainer(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self.model_id = 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
        self.dataset_id = 'speech_asr_aishell1_trainsets'
        self.dataset_namespace = 'speech_asr'

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_trainer(self):
        ds_dict = MsDataset.load(
            self.dataset_id, namespace=self.dataset_namespace)
        kwargs = dict(
            model=self.model_id, work_dir=self.tmp_dir, data_dir=ds_dict)
        trainer = build_trainer(
            Trainers.speech_asr_trainer, default_args=kwargs)
        trainer.train()
        result_model = os.path.join(self.tmp_dir, 'valid.acc.best.pth')
        assert os.path.exists(result_model)


if __name__ == '__main__':
    unittest.main()
