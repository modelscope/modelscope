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


@unittest.skip('For numpy compatible trimesh numpy bool')
class TextureGenerationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.text_texture_generation
        self.model_id = 'damo/cv_diffuser_text-texture-generation'
        self.test_mesh = 'data/test/mesh/texture_generation/mesh1.obj'
        self.prompt = 'old backpack'

    def pipeline_inference(self, pipeline: Pipeline, input_location):
        result = pipeline(input_location)
        mesh = result[OutputKeys.OUTPUT]
        print(f'Output to {osp.abspath("mesh_post.obj")}', mesh)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        model_dir = snapshot_download(self.model_id)
        text_texture_generation = pipeline(
            Tasks.text_texture_generation, model=model_dir)
        input = {
            'mesh_path': self.test_mesh,
            'prompt': self.prompt,
            'image_size': 512,
            'uvsize': 1024
        }
        print('running')
        self.pipeline_inference(text_texture_generation, input)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        text_texture_generation = pipeline(
            Tasks.text_texture_generation, model=self.model_id)
        input = {
            'mesh_path': self.test_mesh,
            'prompt': self.prompt,
            'image_size': 512,
            'uvsize': 1024
        }
        print('running')
        self.pipeline_inference(text_texture_generation, input)


if __name__ == '__main__':
    unittest.main()
