# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.trainers import build_trainer
from modelscope.utils.constant import DownloadMode, Tasks


class TestFinetuneSiameseUIE(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skip(
        'skip since the test requires multiple GPU and takes a long time to run'
    )
    def test_finetune_people_daily(self):
        model_id = 'damo/nlp_structbert_siamese-uie_chinese-base'
        WORK_DIR = '/tmp'
        train_dataset = MsDataset.load(
            'people_daily_ner_1998_tiny',
            namespace='damo',
            split='train',
            download_mode=DownloadMode.FORCE_REDOWNLOAD)
        eval_dataset = MsDataset.load(
            'people_daily_ner_1998_tiny',
            namespace='damo',
            split='validation',
            download_mode=DownloadMode.FORCE_REDOWNLOAD)
        max_epochs = 3
        kwargs = dict(
            model=model_id,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            max_epochs=max_epochs,
            work_dir=WORK_DIR)
        trainer = build_trainer('siamese-uie-trainer', default_args=kwargs)
        trainer.train()
        for i in range(max_epochs):
            eval_results = trainer.evaluate(f'{WORK_DIR}/epoch_{i+1}.pth')
            print(f'epoch {i} evaluation result:')
            print(eval_results)
        pipeline_uie = pipeline(
            task=Tasks.siamese_uie, model=f'{WORK_DIR}/output')
        pipeline_uie(
            input='1944年毕业于北大的名古屋铁道会长谷口清太郎等人在日本积极筹资',
            schema={
                '人物': None,
                '地理位置': None,
                '组织机构': None
            })


if __name__ == '__main__':
    unittest.main()
