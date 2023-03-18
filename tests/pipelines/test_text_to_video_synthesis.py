# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class TextToVideoSynthesisTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.text_to_video_synthesis
        self.model_id = 'damo/text-to-video-synthesis'

    test_text = {
        'text': 'A panda eating bamboo on a rock.',
    }

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        pipe_line_text_to_video_synthesis = pipeline(
            task=self.task, model=self.model_id)
        output_video_path = pipe_line_text_to_video_synthesis(
            self.test_text)[OutputKeys.OUTPUT_VIDEO]
        print(output_video_path)

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
