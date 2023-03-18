# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import unittest

from modelscope.models import Model
from modelscope.models.cv.vidt import VidtModel
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class VidtTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.image_object_detection
        self.model_id = 'damo/ViDT-face-detection'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_pipeline(self):
        vidt_pipeline = pipeline(self.task, self.model_id)
        result = vidt_pipeline('data/test/images/vidt_test1.jpg')
        print(f'Vidt output: {result}.')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_load_model_from_pretrained(self):
        model = Model.from_pretrained('damo/ViDT-face-detection')
        self.assertTrue(model.__class__ == VidtModel)


if __name__ == '__main__':
    unittest.main()
