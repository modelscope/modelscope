# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
import unittest

import cv2

import modelscope
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.cv.image_driving_perception import YOLOPv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.cv import ImageDrivingPerceptionPipeline
from modelscope.preprocessors.image import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import \
    show_image_driving_perception_result
from modelscope.utils.test_utils import test_level


class ImageDrivingPerceptionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_yolopv2_image-driving-perception_bdd100k'
        self.img_path = 'data/test/images/image_driving_perception.jpg'

    def pipeline_inference(self, pipeline: Pipeline, img_path: str):
        result = pipeline(img_path)
        img = LoadImage.convert_to_ndarray(img_path)
        img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)
        show_image_driving_perception_result(
            img, result, out_file='result.jpg', if_draw=[1, 1, 1])
        print(f'Output written to {osp.abspath("result.jpg")}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        image_driving_perception_pipeline = pipeline(
            Tasks.image_driving_perception, model=self.model_id)
        self.pipeline_inference(image_driving_perception_pipeline,
                                self.img_path)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        image_driving_perception_pipeline = pipeline(
            task=Tasks.image_driving_perception, model=model)
        self.pipeline_inference(image_driving_perception_pipeline,
                                self.img_path)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        model = YOLOPv2(cache_path)
        image_driving_perception_pipeline = ImageDrivingPerceptionPipeline(
            model, preprocessor=None)
        self.pipeline_inference(image_driving_perception_pipeline,
                                self.img_path)


if __name__ == '__main__':
    unittest.main()
