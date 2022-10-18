# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest
from typing import List

import json
from transformers import BertTokenizer

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import TableQuestionAnsweringPipeline
from modelscope.preprocessors import TableQuestionAnsweringPreprocessor
from modelscope.preprocessors.star3.fields.database import Database
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.test_utils import test_level


def tableqa_tracking_and_print_results_with_history(
        pipelines: List[TableQuestionAnsweringPipeline]):
    test_case = {
        'utterance': [
            '有哪些风险类型？',
            '风险类型有多少种？',
            '珠江流域的小(2)型水库的库容总量是多少？',
            '那平均值是多少？',
            '那水库的名称呢？',
            '换成中型的呢？',
            '枣庄营业厅的电话',
            '那地址呢？',
            '枣庄营业厅的电话和地址',
        ]
    }
    for p in pipelines:
        historical_queries = None
        for question in test_case['utterance']:
            output_dict = p({
                'question': question,
                'history_sql': historical_queries
            })
            print('question', question)
            print('sql text:', output_dict[OutputKeys.SQL_STRING])
            print('sql query:', output_dict[OutputKeys.SQL_QUERY])
            print('query result:', output_dict[OutputKeys.QUERT_RESULT])
            print('json dumps', json.dumps(output_dict))
            print()
            historical_queries = output_dict[OutputKeys.HISTORY]


def tableqa_tracking_and_print_results_without_history(
        pipelines: List[TableQuestionAnsweringPipeline]):
    test_case = {
        'utterance': [
            '有哪些风险类型？',
            '风险类型有多少种？',
            '珠江流域的小(2)型水库的库容总量是多少？',
            '枣庄营业厅的电话',
            '枣庄营业厅的电话和地址',
        ]
    }
    for p in pipelines:
        for question in test_case['utterance']:
            output_dict = p({'question': question})
            print('question', question)
            print('sql text:', output_dict[OutputKeys.SQL_STRING])
            print('sql query:', output_dict[OutputKeys.SQL_QUERY])
            print('query result:', output_dict[OutputKeys.QUERT_RESULT])
            print('json dumps', json.dumps(output_dict))
            print()


class TableQuestionAnswering(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.table_question_answering
        self.model_id = 'damo/nlp_convai_text2sql_pretrain_cn'

    model_id = 'damo/nlp_convai_text2sql_pretrain_cn'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        preprocessor = TableQuestionAnsweringPreprocessor(model_dir=cache_path)
        pipelines = [
            pipeline(
                Tasks.table_question_answering,
                model=cache_path,
                preprocessor=preprocessor)
        ]
        tableqa_tracking_and_print_results_with_history(pipelines)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        preprocessor = TableQuestionAnsweringPreprocessor(
            model_dir=model.model_dir)
        pipelines = [
            pipeline(
                Tasks.table_question_answering,
                model=model,
                preprocessor=preprocessor)
        ]
        tableqa_tracking_and_print_results_with_history(pipelines)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_task(self):
        pipelines = [pipeline(Tasks.table_question_answering, self.model_id)]
        tableqa_tracking_and_print_results_with_history(pipelines)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_model_from_modelhub_with_other_classes(self):
        model = Model.from_pretrained(self.model_id)
        self.tokenizer = BertTokenizer(
            os.path.join(model.model_dir, ModelFile.VOCAB_FILE))
        db = Database(
            tokenizer=self.tokenizer,
            table_file_path=[
                os.path.join(model.model_dir, 'databases', fname)
                for fname in os.listdir(
                    os.path.join(model.model_dir, 'databases'))
            ],
            syn_dict_file_path=os.path.join(model.model_dir, 'synonym.txt'),
            is_use_sqlite=True)
        preprocessor = TableQuestionAnsweringPreprocessor(
            model_dir=model.model_dir, db=db)
        pipelines = [
            pipeline(
                Tasks.table_question_answering,
                model=model,
                preprocessor=preprocessor,
                db=db)
        ]
        tableqa_tracking_and_print_results_without_history(pipelines)
        tableqa_tracking_and_print_results_with_history(pipelines)


if __name__ == '__main__':
    unittest.main()
