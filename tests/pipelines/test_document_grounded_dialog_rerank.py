# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import json
import torch

from modelscope.models import Model
from modelscope.models.nlp import DocumentGroundedDialogRerankModel
from modelscope.msdatasets import MsDataset
from modelscope.pipelines.nlp import DocumentGroundedDialogRerankPipeline
from modelscope.preprocessors.nlp import \
    DocumentGroundedDialogRerankPreprocessor
from modelscope.utils.constant import DownloadMode, Tasks
from modelscope.utils.test_utils import test_level


class DocumentGroundedDialogRerankTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.document_grounded_dialog_rerank
        self.model_id = 'DAMO_ConvAI/nlp_convai_ranking_pretrain'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run(self):
        args = {
            'output': '../../../result.json',
            'max_batch_size': 64,
            'exclude_instances': '',
            'include_passages': False,
            'do_lower_case': True,
            'max_seq_length': 512,
            'query_length': 195,
            'tokenizer_resize': True,
            'model_resize': True,
            'kilt_data': True
        }
        model = Model.from_pretrained(self.model_id, revision='v1.0.0', **args)
        mypreprocessor = DocumentGroundedDialogRerankPreprocessor(
            model.model_dir, **args)
        pipeline_ins = DocumentGroundedDialogRerankPipeline(
            model=model, preprocessor=mypreprocessor, **args)
        dataset = MsDataset.load(
            'DAMO_ConvAI/FrDoc2BotRerank',
            download_mode=DownloadMode.FORCE_REDOWNLOAD,
            split='test')[:2]
        # print(dataset)
        pipeline_ins(dataset)


if __name__ == '__main__':
    unittest.main()
