# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
import unittest

import cv2

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.cv import DDColorImageColorizationPipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class DDColorImageColorizationTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.image_colorization
        self.model_id = 'damo/cv_ddcolor_image-colorization'
        self.test_image = 'data/test/images/audrey_hepburn.jpg'

    def pipeline_inference(self, pipeline: Pipeline, test_image: str):
        result = pipeline(test_image)
        if result is not None:
            cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
            print(f'Output written to {osp.abspath("result.png")}')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        image_colorization = DDColorImageColorizationPipeline(cache_path)
        self.pipeline_inference(image_colorization, self.test_image)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_pretrained(self):
        model = Model.from_pretrained(self.model_id)
        image_colorization = pipeline(
            task=Tasks.image_colorization, model=model)
        self.pipeline_inference(image_colorization, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        image_colorization = pipeline(
            task=Tasks.image_colorization, model=self.model_id)
        self.pipeline_inference(image_colorization, self.test_image)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        image_colorization = pipeline(Tasks.image_colorization)
        self.pipeline_inference(image_colorization, self.test_image)

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
