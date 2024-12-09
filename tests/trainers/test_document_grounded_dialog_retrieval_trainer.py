# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import json

from modelscope.msdatasets import MsDataset
from modelscope.trainers.nlp.document_grounded_dialog_retrieval_trainer import \
    DocumentGroundedDialogRetrievalTrainer
from modelscope.utils.constant import DownloadMode, ModelFile
from modelscope.utils.test_utils import test_level


class DocumentGroundedDialogRetrievalTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'DAMO_ConvAI/nlp_convai_retrieval_pretrain'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer_with_model_name(self):
        # load data
        train_dataset = MsDataset.load(
            'DAMO_ConvAI/FrDoc2BotRetrieval',
            download_mode=DownloadMode.FORCE_REDOWNLOAD)['train']
        sub_train_dataset = [x for x in train_dataset][:10]
        all_passages = ['阑尾炎', '肠胃炎', '肚脐开始', '肚脐为止']

        trainer = DocumentGroundedDialogRetrievalTrainer(
            model=self.model_id,
            train_dataset=sub_train_dataset,
            eval_dataset=sub_train_dataset,
            all_passages=all_passages)
        trainer.train(
            batch_size=64,
            total_epoches=2,
        )
        trainer.evaluate(
            checkpoint_path=os.path.join(trainer.model.model_dir,
                                         'finetuned_model.bin'))


if __name__ == '__main__':
    unittest.main()
