# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import unittest

import numpy as np
from PIL import Image

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


class ImageTo3DTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'Damo_XR_Lab/Syncdreamer'
        self.input = {
            'input_path': 'data/test/images/basketball.png',
        }

    def pipeline_inference(self, pipeline: Pipeline, input: str):
        result = pipeline(input['input_path'])
        np_content = []
        for idx, img in enumerate(result['MViews']):
            np_content.append(np.array(result['MViews'][idx]))

        np_content = np.concatenate(np_content, axis=1)
        Image.fromarray(np_content).save('./concat.png')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        image_to_3d = pipeline(
            Tasks.image_to_3d, model=self.model_id, revision='v1.0.1')
        self.pipeline_inference(image_to_3d, self.input)


if __name__ == '__main__':
    unittest.main()
