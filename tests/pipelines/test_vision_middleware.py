# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.models import Model
from modelscope.models.cv.vision_middleware import VisionMiddlewareModel
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class VisionMiddlewareTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.image_segmentation
        self.model_id = 'damo/cv_vit-b16_vision-middleware'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_pipeline(self):

        vim_pipeline = pipeline(self.task, self.model_id)
        result = vim_pipeline('data/test/images/vision_middleware_test1.jpg')

        print(f'ViM output: {result}.')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_load_model_from_pretrained(self):
        model = Model.from_pretrained('damo/cv_vit-b16_vision-middleware')
        self.assertTrue(model.__class__ == VisionMiddlewareModel)


if __name__ == '__main__':
    unittest.main()
