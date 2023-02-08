# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
import sys
import unittest

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

    def pipeline_inference(self, pipeline: Pipeline, input_location: str):
        result = pipeline(input_location)
        mesh = result[OutputKeys.OUTPUT]
        write_obj('result_face_reconstruction.obj', mesh)
        print(
            f'Output written to {osp.abspath("result_face_reconstruction.obj")}'
        )

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        model_dir = snapshot_download(self.model_id)
        face_reconstruction = pipeline(
            Tasks.face_reconstruction, model=model_dir)
        self.pipeline_inference(face_reconstruction, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        face_reconstruction = pipeline(
            Tasks.face_reconstruction, model=self.model_id)
        self.pipeline_inference(face_reconstruction, self.test_image)

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
