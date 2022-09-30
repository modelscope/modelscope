# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import unittest

from modelscope.trainers.multi_modal.ofa import OFATrainer
from modelscope.utils.test_utils import test_level


class TestOfaTrainer(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_trainer(self):
        model_id = 'damo/ofa_image-caption_coco_huge_en'
        self.trainer = OFATrainer(model_id)
        os.makedirs(self.trainer.work_dir, exist_ok=True)
        self.trainer.train()
        if os.path.exists(self.trainer.work_dir):
            shutil.rmtree(self.trainer.work_dir)


if __name__ == '__main__':
    unittest.main()
