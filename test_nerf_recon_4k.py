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


class NeRFRecon4KTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'DAMOXR/cv_nerf-3d-reconstruction-4k-nerf_damo'
        data_dir = MsDataset.load(
            'DAMOXR/nerf_llff_data',
            subset_name='default',
            split='test',
            # download_mode=DownloadMode.FORCE_REDOWNLOAD
        ).config_kwargs['split_config']['test']
        nerf_llff = os.path.join(data_dir, 'nerf_llff_data')
        scene = 'fern'
        data_dir = os.path.join(nerf_llff, scene)
        self.render_dir = 'exp'
        self.data_dic = dict(
            datadir=data_dir,
            dataset_type='llff',
            load_sr=1,
            factor=4,
            ndc=True,
            white_bkgd=False)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    @unittest.skipIf(not torch.cuda.is_available(), 'cuda unittest only')
    def test_run_modelhub(self):
        nerf_recon_4k = pipeline(
            Tasks.nerf_recon_4k,
            model=self.model_id,
            data_type='llff',
        )
        nerf_recon_4k(dict(data_cfg=self.data_dic, render_dir=self.render_dir))
        print('4k-nerf_damo.test_run_modelhub done')


if __name__ == '__main__':
    unittest.main()
