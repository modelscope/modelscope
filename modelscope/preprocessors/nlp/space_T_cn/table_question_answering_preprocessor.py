# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict

import torch
from transformers import BertTokenizer

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.preprocessors.nlp.space_T_cn.fields.database import Database
from modelscope.preprocessors.nlp.space_T_cn.fields.schema_link import \
    SchemaLinker
from modelscope.utils.config import Config
from modelscope.utils.constant import Fields, ModelFile
from modelscope.utils.type_assert import type_assert

__all__ = ['TableQuestionAnsweringPreprocessor']


@PREPROCESSORS.register_module(
    Fields.nlp,
    module_name=Preprocessors.table_question_answering_preprocessor)
class TableQuestionAnsweringPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, db: Database = None, *args, **kwargs):
        """preprocess the data

        Args:
            model_dir (str): model path
            db (Database): database instance
        """
        super().__init__(*args, **kwargs)

        self.model_dir: str = model_dir
        self.config = Config.from_file(
            os.path.join(self.model_dir, ModelFile.CONFIGURATION))

        # read tokenizer
        self.tokenizer = BertTokenizer(
            os.path.join(self.model_dir, ModelFile.VOCAB_FILE))

        # read database
        if db is None:
            self.db = Database(
                tokenizer=self.tokenizer,
                table_file_path=os.path.join(self.model_dir, 'table.json'),
                syn_dict_file_path=os.path.join(self.model_dir, 'synonym.txt'))
        else:
            self.db = db

        # get schema linker
        self.schema_linker = SchemaLinker()

        # set device
        self.device = 'cuda' if \
            ('device' not in kwargs or kwargs['device'] == 'gpu') \
            and torch.cuda.is_available() else 'cpu'

    def construct_data(self, search_result_list, nlu, nlu_t, db, history_sql):
        datas = []
        for search_result in search_result_list:
            data = {}
            data['table_id'] = search_result['table_id']
            data['question'] = nlu
            data['question_tok'] = nlu_t
            data['header_tok'] = db.tables[data['table_id']]['header_tok']
            data['types'] = db.tables[data['table_id']]['header_types']
            data['units'] = db.tables[data['table_id']]['header_units']
            data['action'] = 0
            data['sql'] = None
            data['history_sql'] = history_sql
            data['wvi_corenlp'] = []
            data['bertindex_knowledge'] = search_result['question_knowledge']
            data['header_knowledge'] = search_result['header_knowledge']
            data['schema_link'] = search_result['schema_link']
            datas.append(data)

        return datas

    @type_assert(object, dict)
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data (dict):
                utterance: a sentence
                last_sql: predicted sql of last utterance
                Example:
                    utterance: 'Which of these are hiring?'
                    last_sql: ''

        Returns:
            Dict[str, Any]: the preprocessed data
        """

        # tokenize question
        question = data['question']
        table_id = data.get('table_id', None)
        history_sql = data.get('history_sql', None)
        nlu = question.lower()
        nlu_t = self.tokenizer.tokenize(nlu)

        # get linking
        search_result_list = self.schema_linker.get_entity_linking(
            tokenizer=self.tokenizer,
            nlu=nlu,
            nlu_t=nlu_t,
            tables=self.db.tables,
            col_syn_dict=self.db.syn_dict,
            table_id=table_id,
            history_sql=history_sql)

        # collect data
        datas = self.construct_data(
            search_result_list=search_result_list[0:1],
            nlu=nlu,
            nlu_t=nlu_t,
            db=self.db,
            history_sql=history_sql)

        return {'datas': datas}
