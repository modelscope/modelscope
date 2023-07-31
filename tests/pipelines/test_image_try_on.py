# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import unittest

import cv2

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


class ImageTryOnTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_SAL-VTON_virtual-try-on'
        self.input = {
            'person_input_path': 'data/test/images/image_try_on_person.jpg',
            'garment_input_path': 'data/test/images/image_try_on_garment.jpg',
            'mask_input_path': 'data/test/images/image_try_on_mask.jpg'
        }

    def pipeline_inference(self, pipeline: Pipeline, input: str):
        result = pipeline(input)
        logger.info(result)
        cv2.imwrite('result.jpg', result[OutputKeys.OUTPUT_IMG])

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        image_try_on = pipeline(
            Tasks.image_try_on, model=self.model_id, revision='v1.0.1')
        self.pipeline_inference(image_try_on, self.input)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        image_try_on = pipeline(Tasks.image_try_on)
        self.pipeline_inference(image_try_on, self.input)


if __name__ == '__main__':
    unittest.main()
