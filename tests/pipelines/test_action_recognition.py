# Copyright (c) Alibaba, Inc. and its affiliates.
# !/usr/bin/env python
import os.path as osp
import shutil
import tempfile
import unittest

import cv2

from modelscope.fileio import File
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.test_utils import test_level


class ActionRecognitionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_TAdaConv_action-recognition'

    @unittest.skip('deprecated, download model from model hub instead')
    def test_run_with_direct_file_download(self):
        model_path = 'https://aquila2-online-models.oss-cn-shanghai.aliyuncs.com/maas_test/pytorch_model.pt'
        config_path = 'https://aquila2-online-models.oss-cn-shanghai.aliyuncs.com/maas_test/configuration.json'
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_file = osp.join(tmp_dir, ModelFile.TORCH_MODEL_FILE)
            with open(model_file, 'wb') as ofile1:
                ofile1.write(File.read(model_path))
            config_file = osp.join(tmp_dir, ModelFile.CONFIGURATION)
            with open(config_file, 'wb') as ofile2:
                ofile2.write(File.read(config_path))
            recognition_pipeline = pipeline(
                Tasks.action_recognition, model=tmp_dir)
            result = recognition_pipeline(
                'data/test/videos/action_recognition_test_video.mp4')
            print(f'recognition output: {result}.')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        recognition_pipeline = pipeline(
            Tasks.action_recognition, model=self.model_id)
        result = recognition_pipeline(
            'data/test/videos/action_recognition_test_video.mp4')

        print(f'recognition output: {result}.')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        recognition_pipeline = pipeline(Tasks.action_recognition)
        result = recognition_pipeline(
            'data/test/videos/action_recognition_test_video.mp4')

        print(f'recognition output: {result}.')


if __name__ == '__main__':
    unittest.main()
