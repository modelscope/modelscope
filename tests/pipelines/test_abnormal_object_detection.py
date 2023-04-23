# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ObjectDetectionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.image_object_detection
        self.model_id = 'damo/cv_resnet50_object-detection_maskscoring'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_abnormal_object_detection(self):
        input_location = 'data/test/images/image_detection.jpg'
        object_detect = pipeline(self.task, model=self.model_id)
        result = object_detect(input_location)
        print(result)


if __name__ == '__main__':
    unittest.main()
