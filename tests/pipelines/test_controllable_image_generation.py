# Copyright (c) Alibaba, Inc. and its affiliates.
import tempfile
import unittest

import cv2

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.cv import ControllableImageGenerationPipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ControllableImageGenerationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.controllable_image_generation
        self.model_id = 'dienstag/cv_controlnet_controllable-image-generation_nine-annotators'
        self.input = {
            'image':
            'data/test/images/image_inpainting/image_inpainting_mask_1.png',
            'prompt': 'flower'
        }

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        output_image_path = tempfile.NamedTemporaryFile(suffix='.png').name
        pipeline_ins = pipeline(
            self.task, model=self.model_id, control_type='canny')
        output = pipeline_ins(input=self.input)[OutputKeys.OUTPUT_IMG]

        pipeline_ins = pipeline(
            self.task, model=self.model_id, control_type='hough')
        output = pipeline_ins(input=self.input)[OutputKeys.OUTPUT_IMG]

        pipeline_ins = pipeline(
            self.task, model=self.model_id, control_type='hed')
        output = pipeline_ins(input=self.input)[OutputKeys.OUTPUT_IMG]

        pipeline_ins = pipeline(
            self.task, model=self.model_id, control_type='depth')
        output = pipeline_ins(input=self.input)[OutputKeys.OUTPUT_IMG]

        pipeline_ins = pipeline(
            self.task, model=self.model_id, control_type='normal')
        output = pipeline_ins(input=self.input)[OutputKeys.OUTPUT_IMG]

        pipeline_ins = pipeline(
            self.task, model=self.model_id, control_type='pose')
        output = pipeline_ins(input=self.input)[OutputKeys.OUTPUT_IMG]

        pipeline_ins = pipeline(
            self.task, model=self.model_id, control_type='seg')
        output = pipeline_ins(input=self.input)[OutputKeys.OUTPUT_IMG]

        pipeline_ins = pipeline(
            self.task, model=self.model_id, control_type='fake_scribble')
        output = pipeline_ins(input=self.input)[OutputKeys.OUTPUT_IMG]

        pipeline_ins = pipeline(
            self.task, model=self.model_id, control_type='scribble')
        output = pipeline_ins(input=self.input)[OutputKeys.OUTPUT_IMG]
        cv2.imwrite(output_image_path, output)
        print(
            'pipeline: the output image path is {}'.format(output_image_path))


if __name__ == '__main__':
    unittest.main()
