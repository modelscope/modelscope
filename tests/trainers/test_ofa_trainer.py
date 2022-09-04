# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import unittest

from modelscope.trainers.multi_modal.ofa import OFATrainer
from modelscope.utils.test_utils import test_level


class TestOfaTrainer(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer(self):
        model_id = '/apsarapangu/disk2/yichang.zyc/ckpt/MaaS/ofa_text-classification_mnli_large_en'
        self.trainer = OFATrainer(model_id)
        self.trainer.train()
        if os.path.exists(self.trainer.work_dir):
            shutil.rmtree(self.trainer.work_dir)


if __name__ == '__main__':
    unittest.main()
