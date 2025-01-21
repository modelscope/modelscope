# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ObjectDetectionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.human_detection
        self.model_id = 'damo/cv_resnet18_human-detection'

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_object_detection(self):
        input_location = 'data/test/images/image_detection.jpg'
        model_id = 'damo/cv_vit_object-detection_coco'
        object_detect = pipeline(Tasks.image_object_detection, model=model_id)
        result = object_detect(input_location)
        print(result)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_object_detection_with_default_task(self):
        input_location = 'data/test/images/image_detection.jpg'
        object_detect = pipeline(Tasks.image_object_detection)
        result = object_detect(input_location)
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_human_detection(self):
        input_location = 'data/test/images/image_detection.jpg'
        model_id = 'damo/cv_resnet18_human-detection'
        human_detect = pipeline(Tasks.human_detection, model=model_id)
        result = human_detect(input_location)
        print(result)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_human_detection_with_default_task(self):
        input_location = 'data/test/images/image_detection.jpg'
        human_detect = pipeline(Tasks.human_detection)
        result = human_detect(input_location)
        print(result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_image_object_detection_auto_pipeline(self):
        # TODO: to be fixed in the future
        model_id = 'damo/cv_yolox_image-object-detection-auto'
        test_image = 'data/test/images/auto_demo.jpg'

        image_object_detection_auto = pipeline(
            Tasks.image_object_detection, model=model_id)

        result = image_object_detection_auto(test_image)
        image_object_detection_auto.show_result(test_image, result,
                                                'auto_demo_ret.jpg')

    @unittest.skip('skip test in current test level: no pipeline implemented')
    def test_image_object_detection_dino_pipeline(self):
        model_id = 'damo/cv_swinl_image-object-detection_dino'
        test_image = 'data/test/images/image_detection.jpg'
        image_object_detection_dino = pipeline(
            Tasks.image_object_detection, model=model_id)
        result = image_object_detection_dino(test_image)
        print(result)


if __name__ == '__main__':
    unittest.main()
