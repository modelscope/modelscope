# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest
from typing import List

from transformers import BertTokenizer

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import TableQuestionAnsweringPipeline
from modelscope.preprocessors import TableQuestionAnsweringPreprocessor
from modelscope.preprocessors.star3.fields.database import Database
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.nlp.nlp_utils import tableqa_tracking_and_print_results
from modelscope.utils.test_utils import test_level


class TableQuestionAnswering(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.table_question_answering
        self.model_id = 'damo/nlp_convai_text2sql_pretrain_cn'

    model_id = 'damo/nlp_convai_text2sql_pretrain_cn'
    test_case = {
        'utterance':
        ['长江流域的小(2)型水库的库容总量是多少？', '那平均值是多少？', '那水库的名称呢？', '换成中型的呢？']
    }

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        preprocessor = TableQuestionAnsweringPreprocessor(model_dir=cache_path)
        pipelines = [
            TableQuestionAnsweringPipeline(
                model=cache_path, preprocessor=preprocessor)
        ]
        tableqa_tracking_and_print_results(self.test_case, pipelines)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        preprocessor = TableQuestionAnsweringPreprocessor(
            model_dir=model.model_dir)
        pipelines = [
            TableQuestionAnsweringPipeline(
                model=model, preprocessor=preprocessor)
        ]
        tableqa_tracking_and_print_results(self.test_case, pipelines)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_task(self):
        pipelines = [pipeline(Tasks.table_question_answering, self.model_id)]
        tableqa_tracking_and_print_results(self.test_case, pipelines)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_model_from_modelhub_with_other_classes(self):
        model = Model.from_pretrained(self.model_id)
        self.tokenizer = BertTokenizer(
            os.path.join(model.model_dir, ModelFile.VOCAB_FILE))
        db = Database(
            tokenizer=self.tokenizer,
            table_file_path=os.path.join(model.model_dir, 'table.json'),
            syn_dict_file_path=os.path.join(model.model_dir, 'synonym.txt'))
        preprocessor = TableQuestionAnsweringPreprocessor(
            model_dir=model.model_dir, db=db)
        pipelines = [
            TableQuestionAnsweringPipeline(
                model=model, preprocessor=preprocessor, db=db)
        ]
        tableqa_tracking_and_print_results(self.test_case, pipelines)


if __name__ == '__main__':
    unittest.main()
