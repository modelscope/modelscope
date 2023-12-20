# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import torch

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.utils.constant import DownloadMode, Tasks
from modelscope.utils.test_utils import test_level

class GaussianSplatting4DTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'Damo_XR_Lab/4DGaussian_Splatting_for_Real-Time_Dynamic_Scene_Rendering'
        data_dir = MsDataset.load(
            'D_NeRF_Dataset',
            namespace='Damo_XR_Lab',
            split='train',
            download_mode=DownloadMode.FORCE_REDOWNLOAD
        ).config_kwargs['split_config']['train']
        self.source_dir = os.path.join(data_dir, 'data') 


    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    @unittest.skipIf(not torch.cuda.is_available(), 'cuda unittest only')
    def test_run(self):
        snapshot_path = snapshot_download(self.model_id)
        print('snapshot_path: {}'.format(snapshot_path))
        gaussian_splatting_4D = pipeline(
            task=Tasks.gaussian_splatting_4D,
            model=self.model_id
            # ,config_file = os.path.join(modelPath, "configuration.json")
        )

        gaussian_splatting_4D(dict(model_dir=snapshot_path, source_dir = self.source_dir))
        print('gaussian-splatting-4D_damo.test_run_modelhub done')


if __name__ == '__main__':
    unittest.main()
