# Copyright (c) Alibaba, Inc. and its affiliates.
import io
import os
import os.path as osp
import sys
import unittest

import cv2
from moviepy.editor import ImageSequenceClip

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models.cv.face_reconstruction.utils import write_obj
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level

sys.path.append('.')


class FaceReconstructionTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.face_reconstruction
        self.model_id = 'damo/cv_resnet50_face-reconstruction'
        self.test_image = 'data/test/images/face_reconstruction.jpg'

    def save_results(self, result, save_root):
        os.makedirs(save_root, exist_ok=True)

        # export obj and texture
        mesh = result[OutputKeys.OUTPUT]['mesh']
        texture_map = result[OutputKeys.OUTPUT_IMG]
        mesh['texture_map'] = texture_map
        write_obj(os.path.join(save_root, 'hrn_mesh_mid.obj'), mesh)

        # export rotation video
        frame_list = result[OutputKeys.OUTPUT]['frame_list']
        video = ImageSequenceClip(sequence=frame_list, fps=30)
        video.write_videofile(
            os.path.join(save_root, 'rotate.mp4'), fps=30, audio=False)
        del frame_list

        # save visualization image
        vis_image = result[OutputKeys.OUTPUT]['vis_image']
        cv2.imwrite(os.path.join(save_root, 'vis_image.jpg'), vis_image)

        print(f'Output written to {osp.abspath(save_root)}')

    def pipeline_inference(self, pipeline: Pipeline, input_location: str):
        result = pipeline(input_location)
        self.save_results(result, './face_reconstruction_results')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        model_dir = snapshot_download(self.model_id, revision='v2.0.0-HRN')
        face_reconstruction = pipeline(
            Tasks.face_reconstruction, model=model_dir)
        self.pipeline_inference(face_reconstruction, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        face_reconstruction = pipeline(
            Tasks.face_reconstruction,
            model=self.model_id,
            model_revision='v2.0.0-HRN')
        self.pipeline_inference(face_reconstruction, self.test_image)

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
