# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest
from threading import Thread
from typing import List

import json
from transformers import BertTokenizer

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import TableQuestionAnsweringPipeline
from modelscope.preprocessors import TableQuestionAnsweringPreprocessor
from modelscope.preprocessors.nlp.space_T_cn.fields.database import Database
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.test_utils import test_level


def tableqa_tracking_and_print_results_with_history(
        pipelines: List[TableQuestionAnsweringPipeline]):
    test_case = {
        'utterance': [
            '有哪些风险类型？',
            '风险类型有多少种？',
            '珠江流域的小型水库的库容总量是多少？',
            '那平均值是多少？',
            '那水库的名称呢？',
            '换成中型的呢？',
        ]
    }
    for p in pipelines:
        historical_queries = None
        for question in test_case['utterance']:
            output_dict = p({
                'question': question,
                'history_sql': historical_queries
            })[OutputKeys.OUTPUT]
            print('question', question)
            print('sql text:', output_dict[OutputKeys.SQL_STRING])
            print('sql query:', output_dict[OutputKeys.SQL_QUERY])
            print('query result:', output_dict[OutputKeys.QUERY_RESULT])
            print('json dumps', json.dumps(output_dict, ensure_ascii=False))
            print()
            historical_queries = output_dict[OutputKeys.HISTORY]


def tableqa_tracking_and_print_results_without_history(
        pipelines: List[TableQuestionAnsweringPipeline]):
    test_case = {
        'utterance': [['列出油耗大于8但是功率低于200的名称和价格', 'car'],
                      ['油耗低于5的suv有哪些？', 'car'], ['上个月收益超过3的有几个基金？', 'fund'],
                      ['净值不等于1的基金平均月收益率和年收益率是多少？', 'fund'],
                      ['计算机或者成绩优秀的同学有哪些？学号是多少？', 'student'],
                      ['本部博士生中平均身高是多少？', 'student'],
                      ['长江流域和珠江流域的水库库容总量是多少？', 'reservoir'],
                      ['今天星期几？', 'reservoir']]
    }
    for p in pipelines:
        for question, table_id in test_case['utterance']:
            output_dict = p({
                'question': question,
                'table_id': table_id
            })[OutputKeys.OUTPUT]
            print('question', question)
            print('sql text:', output_dict[OutputKeys.SQL_STRING])
            print('sql query:', output_dict[OutputKeys.SQL_QUERY])
            print('query result:', output_dict[OutputKeys.QUERY_RESULT])
            print('json dumps', json.dumps(output_dict, ensure_ascii=False))
            print()


def tableqa_tracking_and_print_results_with_tableid(
        pipelines: List[TableQuestionAnsweringPipeline]):
    test_case = {
        'utterance': [
            ['有哪些风险类型？', 'fund', False],
            ['风险类型有多少种？', 'fund', True],
            ['珠江流域的小型水库的库容总量是多少？', 'reservoir', False],
            ['那平均值是多少？', 'reservoir', True],
            ['那水库的名称呢？', 'reservoir', True],
            ['换成中型的呢？', 'reservoir', True],
            ['近7年来车辆的销量趋势？', 'car_sales', False],
            ['近7年来车辆的销量月环比是多少呢？', 'car_sales', True],
        ],
    }
    for p in pipelines:
        historical_queries = None
        for question, table_id, use_history in test_case['utterance']:
            output_dict = p({
                'question':
                question,
                'table_id':
                table_id,
                'history_sql':
                historical_queries if use_history else None
            })[OutputKeys.OUTPUT]
            print('question', question)
            print('sql text:', output_dict[OutputKeys.SQL_STRING])
            print('sql query:', output_dict[OutputKeys.SQL_QUERY])
            print('query result:', output_dict[OutputKeys.QUERY_RESULT])
            print('json dumps', json.dumps(output_dict, ensure_ascii=False))
            print()
            historical_queries = output_dict[OutputKeys.HISTORY]


class TableQuestionAnswering(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.table_question_answering
        self.model_id = 'damo/nlp_convai_text2sql_pretrain_cn'

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

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download_with_multithreads(self):
        cache_path = snapshot_download(self.model_id)
        pl = pipeline(Tasks.table_question_answering, model=cache_path)

        def print_func(pl, i):
            result = pl({
                'question': '上个月收益从低到高排前七的基金的名称和风险等级是什么',
                'table_id': 'fund',
                'history_sql': None
            })
            print(i, result[OutputKeys.OUTPUT][OutputKeys.SQL_QUERY],
                  result[OutputKeys.OUTPUT][OutputKeys.QUERY_RESULT],
                  json.dumps(result))

        procs = []
        for i in range(5):
            proc = Thread(target=print_func, args=(pl, i))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
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
        tableqa_tracking_and_print_results_with_tableid(pipelines)

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


if __name__ == '__main__':
    unittest.main()
