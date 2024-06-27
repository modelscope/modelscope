# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import imageio

from modelscope.models.cv.human3d_animation.utils import write_obj
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class Human3DRenderTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_3d-human-synthesis-library'
        self.task = Tasks.human3d_render

    def save_results(self, result, save_root):
        os.makedirs(save_root, exist_ok=True)

        mesh = result[OutputKeys.OUTPUT]['mesh']
        write_obj(os.path.join(save_root, 'mesh.obj'), mesh)

        frames_color = result[OutputKeys.OUTPUT]['frames_color']
        imageio.mimwrite(
            os.path.join(save_root, 'render_color.gif'),
            frames_color,
            duration=33)
        del frames_color

        frames_normals = result[OutputKeys.OUTPUT]['frames_normal']
        imageio.mimwrite(
            os.path.join(save_root, 'render_normals.gif'),
            frames_normals,
            duration=33)
        del frames_normals

        print(f'Output written to {os.path.abspath(save_root)}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        human3d = pipeline(self.task, model=self.model_id)
        input = {
            'dataset_id': 'damo/3DHuman_synthetic_dataset',
            # 'case_id': '3f2a7538253e42a8',
            'case_id': '000039',
            'resolution': 1024,
        }
        output = human3d(input)
        self.save_results(output, './human3d_results')

        print('human3d_render.test_run_modelhub done')


if __name__ == '__main__':
    unittest.main()
