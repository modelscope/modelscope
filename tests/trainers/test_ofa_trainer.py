# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import os
import os.path as osp
import shutil
import tempfile
import unittest

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import DownloadMode
from modelscope.utils.test_utils import test_level


class TestOfaTrainer(unittest.TestCase):

    def setUp(self):
        column_map = {'premise': 'text', 'hypothesis': 'text2'}
        data_train = MsDataset.load(
            dataset_name='glue',
            subset_name='mnli',
            namespace='modelscope',
            split='train[:100]',
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
        self.train_dataset = MsDataset.from_hf_dataset(
            data_train._hf_ds.rename_columns(column_map))
        data_eval = MsDataset.load(
            dataset_name='glue',
            subset_name='mnli',
            namespace='modelscope',
            split='validation_matched[:8]',
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
        self.test_dataset = MsDataset.from_hf_dataset(
            data_eval._hf_ds.rename_columns(column_map))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_trainer(self):
        os.environ['LOCAL_RANK'] = '0'
        model_id = 'damo/ofa_text-classification_mnli_large_en'

        kwargs = dict(
            model=model_id,
            cfg_file=
            '/Users/running_you/.cache/modelscope/hub/damo/ofa_text-classification_mnli_large_en//configuration.json',
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            work_dir='/Users/running_you/.cache/modelscope/hub/work/mnli')

        trainer = build_trainer(name=Trainers.ofa_tasks, default_args=kwargs)
        os.makedirs(trainer.work_dir, exist_ok=True)
        trainer.train()
        assert len(
            glob.glob(osp.join(trainer.work_dir,
                               'best_epoch*_accuracy*.pth'))) == 2
        if os.path.exists(self.trainer.work_dir):
            shutil.rmtree(self.trainer.work_dir)


if __name__ == '__main__':
    unittest.main()
