# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import cv2
from PIL import Image

from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import draw_pedestrian_attribute
from modelscope.utils.test_utils import test_level


class PedestrianAttributeRecognitionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.pedestrian_attribute_recognition
        self.model_id = 'damo/cv_resnet50_pedestrian-attribute-recognition_image'
        self.test_image = 'data/test/images/keypoints_detect/000000442836.jpg'

    def pipeline_inference(self, pipeline: Pipeline, pipeline_input):
        output = pipeline(pipeline_input)
        image = draw_pedestrian_attribute(output, self.test_image)
        cv2.imwrite('pedestrian_attribute.jpg', image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub_with_image_file(self):
        pedestrian_attribute_recognition = pipeline(
            self.task, model=self.model_id)
        self.pipeline_inference(pedestrian_attribute_recognition,
                                self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub_with_image_input(self):
        pedestrian_attribute_recognition = pipeline(
            self.task, model=self.model_id)
        self.pipeline_inference(pedestrian_attribute_recognition,
                                Image.open(self.test_image))


if __name__ == '__main__':
    unittest.main()
