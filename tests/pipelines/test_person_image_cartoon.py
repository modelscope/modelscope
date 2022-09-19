# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
import unittest

import cv2

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class ImageCartoonTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_unet_person-image-cartoon_compound-models'
        self.model_id_3d = 'damo/cv_unet_person-image-cartoon-3d_compound-models'
        self.model_id_handdrawn = 'damo/cv_unet_person-image-cartoon-handdrawn_compound-models'
        self.model_id_sketch = 'damo/cv_unet_person-image-cartoon-sketch_compound-models'
        self.model_id_artstyle = 'damo/cv_unet_person-image-cartoon-artstyle_compound-models'
        self.task = Tasks.image_portrait_stylization
        self.test_image = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_cartoon.png'

    def pipeline_inference(self, pipeline: Pipeline, input_location: str):
        result = pipeline(input_location)
        if result is not None:
            cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
            print(f'Output written to {osp.abspath("result.png")}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        img_cartoon = pipeline(
            Tasks.image_portrait_stylization, model=self.model_id)
        self.pipeline_inference(img_cartoon, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub_3d(self):
        img_cartoon = pipeline(
            Tasks.image_portrait_stylization, model=self.model_id_3d)
        self.pipeline_inference(img_cartoon, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub_handdrawn(self):
        img_cartoon = pipeline(
            Tasks.image_portrait_stylization, model=self.model_id_handdrawn)
        self.pipeline_inference(img_cartoon, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub_sketch(self):
        img_cartoon = pipeline(
            Tasks.image_portrait_stylization, model=self.model_id_sketch)
        self.pipeline_inference(img_cartoon, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub_artstyle(self):
        img_cartoon = pipeline(
            Tasks.image_portrait_stylization, model=self.model_id_artstyle)
        self.pipeline_inference(img_cartoon, self.test_image)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        img_cartoon = pipeline(Tasks.image_portrait_stylization)
        self.pipeline_inference(img_cartoon, self.test_image)

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
