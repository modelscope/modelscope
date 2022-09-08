# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import cv2

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import realtime_object_detection_bbox_vis
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class RealtimeObjectDetectionTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_cspnet_image-object-detection_yolox'
        self.model_nano_id = 'damo/cv_cspnet_image-object-detection_yolox_nano_coco'
        self.test_image = 'data/test/images/keypoints_detect/000000438862.jpg'
        self.task = Tasks.image_object_detection

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        realtime_object_detection = pipeline(
            Tasks.image_object_detection, model=self.model_id)

        image = cv2.imread(self.test_image)
        result = realtime_object_detection(image)
        if result:
            bboxes = result[OutputKeys.BOXES].astype(int)
            image = realtime_object_detection_bbox_vis(image, bboxes)
            cv2.imwrite('rt_obj_out.jpg', image)
        else:
            raise ValueError('process error')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_nano(self):
        realtime_object_detection = pipeline(
            Tasks.image_object_detection, model=self.model_nano_id)

        image = cv2.imread(self.test_image)
        result = realtime_object_detection(image)
        if result:
            bboxes = result[OutputKeys.BOXES].astype(int)
            image = realtime_object_detection_bbox_vis(image, bboxes)
            cv2.imwrite('rtnano_obj_out.jpg', image)
        else:
            raise ValueError('process error')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
