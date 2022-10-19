# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import os
import os.path as osp
import shutil
import unittest

from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.utils.test_utils import test_level


class TestOfaTrainer(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_trainer(self):
        os.environ['LOCAL_RANK'] = '0'
        model_id = 'damo/ofa_text-classification_mnli_large_en'
        default_args = {'model': model_id}
        trainer = build_trainer(
            name=Trainers.ofa_tasks, default_args=default_args)
        os.makedirs(trainer.work_dir, exist_ok=True)
        trainer.train()
        assert len(
            glob.glob(osp.join(trainer.work_dir,
                               'best_epoch*_accuracy*.pth'))) == 2
        if os.path.exists(self.trainer.work_dir):
            shutil.rmtree(self.trainer.work_dir)


if __name__ == '__main__':
    unittest.main()
