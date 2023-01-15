# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import numpy as np
import torch

from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class MultiStageDiffusionTest(unittest.TestCase):
    model_id = 'damo/cv_diffusion_text-to-image-synthesis'
    test_text = {'text': 'Photograph of a baby chicken wearing sunglasses'}

    @unittest.skip(
        'skip test since the pretrained model is not publicly available')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        pipe_line_text_to_image_synthesis = pipeline(
            task=Tasks.text_to_image_synthesis, model=model)
        img = pipe_line_text_to_image_synthesis(
            self.test_text)[OutputKeys.OUTPUT_IMGS][0]
        print(np.sum(np.abs(img)))

    @unittest.skip(
        'skip test since the pretrained model is not publicly available')
    def test_run_with_model_name(self):
        pipe_line_text_to_image_synthesis = pipeline(
            task=Tasks.text_to_image_synthesis, model=self.model_id)
        img = pipe_line_text_to_image_synthesis(
            self.test_text)[OutputKeys.OUTPUT_IMGS][0]
        print(np.sum(np.abs(img)))


if __name__ == '__main__':
    unittest.main()
