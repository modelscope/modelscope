# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class SurfaceReconCommonTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_surface-reconstruction-common'
        self.task = Tasks.surface_recon_common
        data_dir = MsDataset.load(
            'surface_recon_dataset', namespace='menyifang',
            split='train').config_kwargs['split_config']['train']
        data_dir = os.path.join(data_dir, 'surface_recon_dataset')
        self.data_dir = data_dir
        self.save_dir = '.'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        surface_recon_common = pipeline(
            self.task,
            model=self.model_id,
        )

        surface_recon_common(
            dict(data_dir=self.data_dir, save_dir=self.save_dir))
        print('surface_recon_common.test_run_modelhub done')


if __name__ == '__main__':
    unittest.main()
