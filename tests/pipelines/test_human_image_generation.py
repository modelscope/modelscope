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


class HumanImageGenerationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_FreqHPT_human-image-generation'
        self.input = {
            'source_img_path':
            'data/test/images/human_image_generation_source_img.jpg',
            'target_pose_path':
            'data/test/images/human_image_generation_target_pose.txt'
        }

    def pipeline_inference(self, pipeline: Pipeline, input: str):
        result = pipeline(input)
        logger.info(result)
        cv2.imwrite('result.jpg', result[OutputKeys.OUTPUT_IMG])

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        human_image_generation = pipeline(
            Tasks.human_image_generation,
            model=self.model_id,
            revision='v1.0.1')
        self.pipeline_inference(human_image_generation, self.input)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        human_image_generation = pipeline(Tasks.human_image_generation)
        self.pipeline_inference(human_image_generation, self.input)


if __name__ == '__main__':
    unittest.main()
