# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import json

from modelscope.msdatasets import MsDataset
from modelscope.trainers.nlp.table_question_answering_trainer import \
    TableQuestionAnsweringTrainer
from modelscope.utils.constant import DownloadMode, ModelFile
from modelscope.utils.test_utils import test_level


class TableQuestionAnsweringTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer_with_model_name(self):
        # load data
        input_dataset = MsDataset.load(
            'ChineseText2SQL', download_mode=DownloadMode.FORCE_REDOWNLOAD)
        train_dataset = []
        for name in input_dataset['train']._hf_ds.data[1]:
            train_dataset.append(json.load(open(str(name), 'r')))
        eval_dataset = []
        for name in input_dataset['test']._hf_ds.data[1]:
            eval_dataset.append(json.load(open(str(name), 'r')))
        print('size of training set', len(train_dataset))
        print('size of evaluation set', len(eval_dataset))

        model_id = 'damo/nlp_convai_text2sql_pretrain_cn'
        trainer = TableQuestionAnsweringTrainer(
            model=model_id,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        trainer.train(
            batch_size=8,
            total_epoches=2,
        )
        trainer.evaluate(
            checkpoint_path=os.path.join(trainer.model.model_dir,
                                         'finetuned_model.bin'))


if __name__ == '__main__':
    unittest.main()
