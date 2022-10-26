# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import shutil
import tempfile
import unittest
from functools import partial

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.audio.audio_utils import to_segment
from modelscope.utils.hub import read_config
from modelscope.utils.test_utils import test_level

SEGMENT_LENGTH_TEST = 640


class TestANSTrainer(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self.model_id = 'damo/speech_frcrn_ans_cirm_16k'
        cfg = read_config(self.model_id)
        cfg.train.max_epochs = 2
        cfg.train.dataloader.batch_size_per_gpu = 1
        self.cfg_file = os.path.join(self.tmp_dir, 'train_config.json')
        cfg.dump(self.cfg_file)

        hf_ds = MsDataset.load(
            'ICASSP_2021_DNS_Challenge', split='test').to_hf_dataset()
        mapped_ds = hf_ds.map(
            partial(to_segment, segment_length=SEGMENT_LENGTH_TEST),
            remove_columns=['duration'],
            batched=True,
            batch_size=2)
        self.dataset = MsDataset.from_hf_dataset(mapped_ds)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer(self):
        kwargs = dict(
            model=self.model_id,
            train_dataset=self.dataset,
            eval_dataset=self.dataset,
            max_epochs=2,
            train_iters_per_epoch=2,
            val_iters_per_epoch=1,
            cfg_file=self.cfg_file,
            work_dir=self.tmp_dir)

        trainer = build_trainer(
            Trainers.speech_frcrn_ans_cirm_16k, default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(2):
            self.assertIn(f'epoch_{i + 1}.pth', results_files)
