# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import cv2

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import realtime_object_detection_bbox_vis
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


class RealtimeObjectDetectionTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.easycv_small_model_id = 'damo/cv_cspnet_image-object-detection_yolox'
        self.easycv_nano_model_id = 'damo/cv_cspnet_image-object-detection_yolox_nano_coco'
        self.test_image = 'data/test/images/keypoints_detect/000000438862.jpg'
        self.task = Tasks.image_object_detection

    @unittest.skip('skip test in current test level: no pipeline implemented')
    def test_run_easycv_yolox(self):
        realtime_object_detection = pipeline(
            Tasks.image_object_detection, model=self.easycv_small_model_id)

        image = cv2.imread(self.test_image)
        result = realtime_object_detection(image)
        if result:
            logger.info(result)
        else:
            raise ValueError('process error')

    @unittest.skip('skip test in current test level: no pipeline implemented')
    def test_run_easycv_yolox_nano(self):
        realtime_object_detection = pipeline(
            Tasks.image_object_detection, model=self.easycv_nano_model_id)

        image = cv2.imread(self.test_image)
        result = realtime_object_detection(image)
        if result:
            logger.info(result)
        else:
            raise ValueError('process error')

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
