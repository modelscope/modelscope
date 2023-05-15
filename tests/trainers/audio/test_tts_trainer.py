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


class TestTtsTrainer(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self.model_id = 'speech_tts/speech_sambert-hifigan_tts_zh-cn_multisp_pretrain_16k'
        self.dataset_id = 'speech_kantts_opendata'
        self.dataset_namespace = 'speech_tts'
        self.train_info = {
            TtsTrainType.TRAIN_TYPE_SAMBERT: {
                'train_steps': 2,
                'save_interval_steps': 1,
                'eval_interval_steps': 1,
                'log_interval': 1
            },
            TtsTrainType.TRAIN_TYPE_VOC: {
                'train_steps': 2,
                'save_interval_steps': 1,
                'eval_interval_steps': 1,
                'log_interval': 1
            }
        }

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_trainer(self):
        kwargs = dict(
            model=self.model_id,
            work_dir=self.tmp_dir,
            train_dataset=self.dataset_id,
            train_dataset_namespace=self.dataset_namespace,
            train_type=self.train_info)
        trainer = build_trainer(
            Trainers.speech_kantts_trainer, default_args=kwargs)
        trainer.train()
        tmp_am = os.path.join(self.tmp_dir, 'tmp_am', 'ckpt')
        tmp_voc = os.path.join(self.tmp_dir, 'tmp_voc', 'ckpt')
        assert os.path.exists(tmp_am)
        assert os.path.exists(tmp_voc)


if __name__ == '__main__':
    unittest.main()
