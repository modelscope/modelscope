# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import unittest

from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class HandStaticTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model = 'damo/cv_mobileface_hand-static'
        self.input = {'img_path': 'data/test/images/hand_static.jpg'}

    def pipeline_inference(self, pipeline: Pipeline, input: str):
        result = pipeline(input)
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        hand_static = pipeline(Tasks.hand_static, model=self.model)
        self.pipeline_inference(hand_static, self.input)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        hand_static = pipeline(Tasks.hand_static)
        self.pipeline_inference(hand_static, self.input)


if __name__ == '__main__':
    unittest.main()
