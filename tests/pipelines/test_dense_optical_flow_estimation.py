# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import cv2
import numpy as np

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

class DenseOpticalFlowEstimationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = 'dense-optical-flow-estimation'
        self.model_id = 'Damo_XR_Lab/cv_raft_dense-optical-flow_things'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_dense_optical_flow_estimation(self):
        input_location = [[
            'data/test/images/dense_flow1.png',
            'data/test/images/dense_flow2.png'
        ]]
        estimator = pipeline(Tasks.dense_optical_flow_estimation, model=self.model_id)
        result = estimator(input_location)
        print(type(result[0]), result[0].keys())
        flow = result[0][OutputKeys.FLOWS]
        flow_vis = result[0][OutputKeys.FLOWS_COLOR]
        cv2.imwrite('result.jpg', flow_vis)

        print('test_dense_optical_flow_estimation DONE')


if __name__ == '__main__':
    unittest.main()
