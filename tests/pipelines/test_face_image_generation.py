# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
import unittest

import cv2

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class FaceGenerationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_gan_face-image-generation'

    def pipeline_inference(self, pipeline: Pipeline, seed: int):
        result = pipeline(seed)
        if result is not None:
            cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
            print(f'Output written to {osp.abspath("result.png")}')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_modelhub(self):
        seed = 10
        face_generation = pipeline(
            Tasks.face_image_generation,
            model=self.model_id,
        )
        self.pipeline_inference(face_generation, seed)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        seed = 10
        face_generation = pipeline(Tasks.face_image_generation)
        self.pipeline_inference(face_generation, seed)


if __name__ == '__main__':
    unittest.main()
