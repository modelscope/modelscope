# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import cv2
import torch

from modelscope import get_logger
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.msdatasets import MsDataset
from modelscope.outputs.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import DownloadMode, Tasks
from modelscope.utils.test_utils import test_level

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
logger = get_logger()


class SelfSupervisedDepthCompletionTest(unittest.TestCase):
    """class SelfSupervisedDepthCompletionTest"""

    def setUp(self) -> None:
        self.model_id = 'Damo_XR_Lab/Self_Supervised_Depth_Completion'
        data_dir = MsDataset.load(
            'KITTI_Depth_Dataset',
            namespace='Damo_XR_Lab',
            split='test',
            download_mode=DownloadMode.FORCE_REDOWNLOAD
        ).config_kwargs['split_config']['test']
        self.source_dir = os.path.join(data_dir, 'selected_data')
        logger.info(data_dir)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    @unittest.skipIf(not torch.cuda.is_available(), 'cuda unittest only')
    def test_run(self):
        """test running evaluation"""
        snapshot_path = snapshot_download(self.model_id)
        logger.info('snapshot_path: %s', snapshot_path)
        self_supervised_depth_completion = pipeline(
            task=Tasks.self_supervised_depth_completion,
            model=self.model_id
            # ,config_file = os.path.join(modelPath, "configuration.json")
        )

        result = self_supervised_depth_completion(
            dict(model_dir=snapshot_path, source_dir=self.source_dir))
        cv2.imwrite('result.jpg', result[OutputKeys.OUTPUT])
        logger.info(
            'self-supervised-depth-completion_damo.test_run_modelhub done')


if __name__ == '__main__':
    unittest.main()
