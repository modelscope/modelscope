# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
import sys
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

sys.path.append('.')


class HumanReconstructionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.human_reconstruction
        self.model_id = 'damo/cv_hrnet_image-human-reconstruction'
        self.test_image = 'data/test/images/human_reconstruction.jpg'

    def pipeline_inference(self, pipeline: Pipeline, input_location: str):
        result = pipeline(input_location)
        mesh = result[OutputKeys.OUTPUT]
        print(
            f'Output to {osp.abspath("human_reconstruction.obj")}, vertices num: {mesh["vertices"].shape}'
        )

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        model_dir = snapshot_download(self.model_id)
        human_reconstruction = pipeline(
            Tasks.human_reconstruction, model=model_dir)
        print('running')
        self.pipeline_inference(human_reconstruction, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        human_reconstruction = pipeline(
            Tasks.human_reconstruction, model=self.model_id)
        self.pipeline_inference(human_reconstruction, self.test_image)


if __name__ == '__main__':
    unittest.main()
