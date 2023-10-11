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


class HeadReconstructionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.head_reconstruction
        self.model_id = 'damo/cv_HRN_head-reconstruction'
        self.test_image = 'data/test/images/face_reconstruction.jpg'

    def save_results(self, result, save_root):
        os.makedirs(save_root, exist_ok=True)

        # export obj and texture
        mesh = result[OutputKeys.OUTPUT]['mesh']
        texture_map = result[OutputKeys.OUTPUT_IMG]
        mesh['texture_map'] = texture_map
        write_obj(os.path.join(save_root, 'head_recon_result.obj'), mesh)

        print(f'Output written to {osp.abspath(save_root)}')

    def pipeline_inference(self, pipeline: Pipeline, input_location: str):
        result = pipeline(input_location)
        self.save_results(result, './head_reconstruction_results')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        model_dir = snapshot_download(self.model_id, revision='v0.2')
        head_reconstruction = pipeline(
            Tasks.head_reconstruction, model=model_dir)
        self.pipeline_inference(head_reconstruction, self.test_image)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub(self):
        head_reconstruction = pipeline(
            Tasks.head_reconstruction,
            model=self.model_id,
            model_revision='v0.2')
        self.pipeline_inference(head_reconstruction, self.test_image)


if __name__ == '__main__':
    unittest.main()
