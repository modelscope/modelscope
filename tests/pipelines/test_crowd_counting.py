# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import cv2
from PIL import Image

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import numpy_to_cv2img
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


class CrowdCountingTest(unittest.TestCase):

    def setUp(self) -> None:
        self.input_location = 'data/test/images/crowd_counting.jpg'
        self.model_id = 'damo/cv_hrnet_crowd-counting_dcanet'

    def save_result(self, result):
        print('scores:', result[OutputKeys.SCORES])
        vis_img = result[OutputKeys.OUTPUT_IMG]
        vis_img = numpy_to_cv2img(vis_img)
        cv2.imwrite('result.jpg', vis_img)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_crowd_counting(self):
        crowd_counting = pipeline(Tasks.crowd_counting, model=self.model_id)
        result = crowd_counting(self.input_location)
        if result:
            self.save_result(result)
        else:
            raise ValueError('process error')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_crowd_counting_with_image(self):
        crowd_counting = pipeline(Tasks.crowd_counting, model=self.model_id)
        img = Image.open(self.input_location)
        result = crowd_counting(img)
        if result:
            self.save_result(result)
        else:
            raise ValueError('process error')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_crowd_counting_with_default_task(self):
        crowd_counting = pipeline(Tasks.crowd_counting)
        result = crowd_counting(self.input_location)
        if result:
            self.save_result(result)
        else:
            raise ValueError('process error')


if __name__ == '__main__':
    unittest.main()
