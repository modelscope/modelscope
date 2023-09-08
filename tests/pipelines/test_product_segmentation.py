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


class ProductSegmentationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_F3Net_product-segmentation'
        self.input = 'data/test/images/product_segmentation.jpg'

    def pipeline_inference(self, pipeline: Pipeline, input: str):
        result = pipeline(input)
        cv2.imwrite('test_product_segmentation_mask.jpg',
                    result[OutputKeys.MASKS])
        logger.info('test done')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        product_segmentation = pipeline(
            Tasks.product_segmentation, model=self.model_id)
        self.pipeline_inference(product_segmentation, self.input)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        product_segmentation = pipeline(Tasks.product_segmentation)
        self.pipeline_inference(product_segmentation, self.input)


if __name__ == '__main__':
    unittest.main()
