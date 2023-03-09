# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import cv2

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class DiscoGuidedDiffusionTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.text_to_image_synthesis
        self.model_id1 = 'yyqoni/yinyueqin_test'
        self.model_id2 = 'yyqoni/yinyueqin_cyberpunk'

    test_input1 = '夕阳西下'
    test_input2 = '城市，赛博朋克'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run(self):
        diffusers_pipeline = pipeline(
            task=self.task, model=self.model_id1, model_revision='v1.0')
        output = diffusers_pipeline({
            'text': self.test_input1,
            'height': 256,
            'width': 256
        })
        cv2.imwrite('output1.png', output['output_imgs'][0])
        print('Image saved to output1.png')

        diffusers_pipeline = pipeline(
            task=self.task, model=self.model_id2, model_revision='v1.0')
        output = diffusers_pipeline({
            'text': self.test_input2,
            'height': 256,
            'width': 256
        })
        cv2.imwrite('output2.png', output['output_imgs'][0])
        print('Image saved to output2.png')


if __name__ == '__main__':
    unittest.main()
