# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class Human3DAnimationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_3d-human-animation'
        self.task = Tasks.human3d_animation

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        human3d = pipeline(self.task, model=self.model_id)
        input = {
            'dataset_id': 'damo/3DHuman_synthetic_dataset',
            'case_id': '3f2a7538253e42a8',
            'action_dataset': 'damo/3DHuman_action_dataset',
            'action': 'SwingDancing',
            'save_dir': 'outputs',
        }
        output = human3d(input)
        print('saved animation file to %s' % output)

        print('human3d_animation.test_run_modelhub done')


if __name__ == '__main__':
    unittest.main()
