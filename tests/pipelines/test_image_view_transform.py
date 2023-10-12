# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import unittest

import cv2
import numpy as np
import torch
from PIL import Image

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


class ImageViewTransformTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_image-view-transform'
        image = Image.open(
            'data/test/images/image_view_transform_source_img.png')
        self.input = {
            'source_img': image,
            'target_view': [50.0, 0.0, 0.0, True, 3.0, 4, 50, 1.0]
        }

    def pipeline_inference(self, pipeline: Pipeline, input: str):
        result = pipeline(input)
        logger.info(result)
        cv2.imwrite('result.jpg', result[OutputKeys.OUTPUT_IMGS][0])
        print(np.sum(np.abs(result[OutputKeys.OUTPUT_IMGS][0])))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        image_view_transform = pipeline(
            Tasks.image_view_transform, model=self.model_id, revision='v1.0.3')
        self.pipeline_inference(image_view_transform, self.input)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        image_view_transform = pipeline(Tasks.image_view_transform)
        self.pipeline_inference(image_view_transform, self.input)


if __name__ == '__main__':
    unittest.main()
