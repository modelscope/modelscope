# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ObjectDetectionTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_object_detection(self):
        input_location = 'data/test/images/image_detection.jpg'
        model_id = 'damo/cv_vit_object-detection_coco'
        object_detect = pipeline(Tasks.image_object_detection, model=model_id)
        result = object_detect(input_location)
        if result:
            print(result)
        else:
            raise ValueError('process error')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_object_detection_with_default_task(self):
        input_location = 'data/test/images/image_detection.jpg'
        object_detect = pipeline(Tasks.image_object_detection)
        result = object_detect(input_location)
        if result:
            print(result)
        else:
            raise ValueError('process error')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_human_detection(self):
        input_location = 'data/test/images/image_detection.jpg'
        model_id = 'damo/cv_resnet18_human-detection'
        human_detect = pipeline(Tasks.human_detection, model=model_id)
        result = human_detect(input_location)
        if result:
            print(result)
        else:
            raise ValueError('process error')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_human_detection_with_default_task(self):
        input_location = 'data/test/images/image_detection.jpg'
        human_detect = pipeline(Tasks.human_detection)
        result = human_detect(input_location)
        if result:
            print(result)
        else:
            raise ValueError('process error')


if __name__ == '__main__':
    unittest.main()
