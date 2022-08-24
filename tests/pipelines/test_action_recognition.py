# Copyright (c) Alibaba, Inc. and its affiliates.
# !/usr/bin/env python
import os.path as osp
import tempfile
import unittest

from modelscope.fileio import File
from modelscope.pipelines import pipeline
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.test_utils import test_level


class ActionRecognitionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_TAdaConv_action-recognition'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        recognition_pipeline = pipeline(
            Tasks.action_recognition, model=self.model_id)
        result = recognition_pipeline(
            'data/test/videos/action_recognition_test_video.mp4')

        print(f'recognition output: {result}.')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        recognition_pipeline = pipeline(Tasks.action_recognition)
        result = recognition_pipeline(
            'data/test/videos/action_recognition_test_video.mp4')

        print(f'recognition output: {result}.')


if __name__ == '__main__':
    unittest.main()
