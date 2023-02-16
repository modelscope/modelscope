# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import numpy as np

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class PointCloudSceneFlowEstimationTest(unittest.TestCase,
                                        DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = 'pointcloud-sceneflow-estimation'
        self.model_id = 'damo/cv_pointnet2_sceneflow-estimation_general'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_pointcloud_scenelfow_estimation(self):
        input_location = ('data/test/pointclouds/flyingthings_pcd1.npy',
                          'data/test/pointclouds/flyingthings_pcd2.npy')
        estimator = pipeline(
            Tasks.pointcloud_sceneflow_estimation, model=self.model_id)
        result = estimator(input_location)
        flow = result[OutputKeys.OUTPUT]
        pcd12 = result[OutputKeys.PCD12]
        pcd12_align = result[OutputKeys.PCD12_ALIGN]

        print(f'pred flow shape:{flow.shape}')
        np.save('flow.npy', flow)
        # visualization
        pcd12.write('pcd12.ply')
        pcd12_align.write('pcd12_align.ply')

        print('test_pointcloud_scenelfow_estimation DONE')


if __name__ == '__main__':
    unittest.main()
