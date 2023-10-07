# Copyright (c) Alibaba, Inc. and its affiliates.
import io
import os
import os.path as osp
import sys
import unittest

import cv2

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models.cv.face_reconstruction.utils import write_obj
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

sys.path.append('.')


class TextToHeadTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.text_to_head
        self.model_id = 'damo/cv_HRN_text-to-head'
        self.test_prompt = 'a clown with red nose'

    def save_results(self, result, save_root):
        os.makedirs(save_root, exist_ok=True)

        # export obj and texture
        mesh = result[OutputKeys.OUTPUT]['mesh']
        texture_map = result[OutputKeys.OUTPUT_IMG]
        mesh['texture_map'] = texture_map
        write_obj(os.path.join(save_root, 'text_to_head_result.obj'), mesh)

        image = result['image']
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(save_root, 'text_to_head_image.jpg'), image_bgr)

        print(f'Output written to {osp.abspath(save_root)}')

    def pipeline_inference(self, pipeline: Pipeline, prompt: str):
        result = pipeline(prompt)
        self.save_results(result, './text_to_head_results')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        model_dir = snapshot_download(self.model_id, revision='v0.1')
        text_to_head = pipeline(Tasks.text_to_head, model=model_dir)
        self.pipeline_inference(text_to_head, self.test_prompt)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub(self):
        face_reconstruction = pipeline(
            Tasks.text_to_head, model=self.model_id, model_revision='v0.1')
        self.pipeline_inference(face_reconstruction, self.test_prompt)


if __name__ == '__main__':
    unittest.main()
