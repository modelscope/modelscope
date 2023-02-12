# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import json

from modelscope.msdatasets import MsDataset
from modelscope.trainers.nlp.document_grounded_dialog_generate_trainer import \
    DocumentGroundedDialogGenerateTrainer
from modelscope.utils.constant import DownloadMode, ModelFile
from modelscope.utils.test_utils import test_level


class DocumentGroundedDialogGenerateTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'DAMO_ConvAI/nlp_convai_generation_pretrain'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer_with_model_name(self):
        # load data
        train_dataset = MsDataset.load(
            'DAMO_ConvAI/FrDoc2BotGeneration',
            download_mode=DownloadMode.FORCE_REDOWNLOAD)
        test_len = 1
        sub_train_dataset = [x for x in train_dataset][:1]
        sub_train_dataset = [{
            'query':
            x['query'][:test_len],
            'rerank':
            json.dumps([p[:test_len] for p in json.loads(x['rerank'])]),
            'response':
            x['response'][:test_len]
        } for x in sub_train_dataset]

        trainer = DocumentGroundedDialogGenerateTrainer(
            model=self.model_id,
            train_dataset=sub_train_dataset,
            eval_dataset=sub_train_dataset,
        )
        trainer.model.model.config['num_beams'] = 1
        trainer.model.model.config['target_sequence_length'] = test_len
        trainer.train(batch_size=1, total_epoches=1, learning_rate=2e-4)
        trainer.evaluate(
            checkpoint_path=os.path.join(trainer.model.model_dir,
                                         'finetuned_model.bin'))


if __name__ == '__main__':
    unittest.main()
