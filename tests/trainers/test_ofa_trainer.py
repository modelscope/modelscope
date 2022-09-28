# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import unittest

from modelscope.trainers.multi_modal.ofa import OFATrainer
from modelscope.utils.test_utils import test_level


class TestOfaTrainer(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer(self):
        model_id = '/apsarapangu/disk2/yichang.zyc/ckpt/MaaS/maas_mnli_pretrain_ckpt'
        self.trainer = OFATrainer(model_id, launcher='pytorch')
        self.trainer.train()
        if os.path.exists(self.trainer.work_dir):
            pass


if __name__ == '__main__':
    unittest.main()
