# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
import unittest

import cv2

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ImageBodyReshapingTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.image_body_reshaping
        self.model_id = 'damo/cv_flow-based-body-reshaping_damo'
        self.test_image = 'data/test/images/image_body_reshaping.jpg'

    def pipeline_inference(self, pipeline: Pipeline, input_location: str):
        result = pipeline(input_location)
        if result is not None:
            cv2.imwrite('result_bodyreshaping.png',
                        result[OutputKeys.OUTPUT_IMG])
            print(
                f'Output written to {osp.abspath("result_body_reshaping.png")}'
            )
        else:
            raise Exception('Testing failed: invalid output')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        model_dir = snapshot_download(self.model_id)
        image_body_reshaping = pipeline(
            Tasks.image_body_reshaping, model=model_dir)
        self.pipeline_inference(image_body_reshaping, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        image_body_reshaping = pipeline(
            Tasks.image_body_reshaping, model=self.model_id)
        self.pipeline_inference(image_body_reshaping, self.test_image)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        image_body_reshaping = pipeline(Tasks.image_body_reshaping)
        self.pipeline_inference(image_body_reshaping, self.test_image)


if __name__ == '__main__':
    unittest.main()
