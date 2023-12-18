# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import torch

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.msdatasets import MsDataset
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import DownloadMode, Tasks
from modelscope.utils.test_utils import test_level


class GaussianSplattingReconTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'Damo_XR_Lab/cv_gaussian-splatting-recon_damo'
        # TODO: test pretrained model path

        self.pretrained_model = 'chair'
        self.data_type = 'blender'
        self.ckpt_path = None
        data_dir = MsDataset.load('nerf_recon_dataset', namespace='damo',
                                  split='train').config_kwargs['split_config']['train']
        nerf_synthetic_dataset = os.path.join(data_dir, 'nerf_synthetic')
        blender_scene = self.pretrained_model
        self.data_dir = os.path.join(nerf_synthetic_dataset, blender_scene)


    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    @unittest.skipIf(not torch.cuda.is_available(), 'cuda unittest only')
    def test_run_modelhub(self):
        # snapshot_path = snapshot_download(self.model_id)
        # print('snapshot_path: {}'.format(snapshot_path))
        # self.ckpt_path = os.path.join(snapshot_path,'pretrained_models', self.pretrained_model)
        # print('ckpt_path: {}',format(self.ckpt_path))
        self.ckpt_path = ''

        gaussian_splatting_recon = pipeline(
            Tasks.gaussian_splatting_recon,
            model=self.model_id,
            data_type=self.data_type,
            data_dir=self.data_dir,
            ckpt_path=self.ckpt_path
        )

        gaussian_splatting_recon(dict(test_mode='evaluation', render_dir=''))
        print('gaussian_splatting_recon.test_run_modelhub done')


if __name__ == '__main__':
    unittest.main()
