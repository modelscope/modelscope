# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
import unittest

import cv2

from modelscope.msdatasets import MsDataset
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ObjectDetection3DTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.object_detection_3d
        self.model_id = 'damo/cv_object-detection-3d_depe'
        ms_ds_nuscenes = MsDataset.load('nuScenes_mini', namespace='shaoxuan')

        data_path = ms_ds_nuscenes.config_kwargs['split_config']
        val_dir = data_path['validation']
        self.val_root = val_dir + '/' + os.listdir(val_dir)[0] + '/'

    def pipeline_inference(self, pipeline: Pipeline, sample_idx: int):
        input_dict = {'data_root': self.val_root, 'sample_idx': sample_idx}

        result = pipeline(input_dict, save_path='./depe_result')
        if result is not None:
            cv2.imwrite('result.jpg', result[OutputKeys.OUTPUT_IMG])
            print(f'Output written to {osp.abspath("result.jpg")}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        idx = 10
        detect = pipeline(
            self.task,
            model=self.model_id,
        )
        self.pipeline_inference(detect, idx)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        idx = 0
        detect = pipeline(self.task)
        self.pipeline_inference(detect, idx)


if __name__ == '__main__':
    unittest.main()
