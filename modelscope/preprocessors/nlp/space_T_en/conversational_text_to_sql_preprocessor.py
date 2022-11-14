# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict

import json
import torch
from text2sql_lgesql.preprocess.graph_utils import GraphProcessor
from text2sql_lgesql.preprocess.process_graphs import process_dataset_graph
from text2sql_lgesql.utils.batch import Batch
from text2sql_lgesql.utils.example import Example

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.preprocessors.nlp.space_T_en.fields import SubPreprocessor
from modelscope.preprocessors.nlp.space_T_en.fields.preprocess_dataset import \
    preprocess_dataset
from modelscope.preprocessors.nlp.space_T_en.fields.process_dataset import (
    process_dataset, process_tables)
from modelscope.utils.config import Config
from modelscope.utils.constant import Fields, ModelFile
from modelscope.utils.type_assert import type_assert

__all__ = ['ConversationalTextToSqlPreprocessor']


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.conversational_text_to_sql)
class ConversationalTextToSqlPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        """preprocess the data

        Args:
            model_dir (str): model path
        """
        super().__init__(*args, **kwargs)

        self.model_dir: str = model_dir

        self.config = Config.from_file(
            os.path.join(self.model_dir, ModelFile.CONFIGURATION))
        self.device = 'cuda' if \
            ('device' not in kwargs or kwargs['device'] == 'gpu') \
            and torch.cuda.is_available() else 'cpu'
        self.processor = None
        self.table_path = os.path.join(self.model_dir, 'tables.json')
        self.tables = json.load(open(self.table_path, 'r', encoding='utf-8'))
        self.output_tables = None
        self.path_cache = []
        self.graph_processor = GraphProcessor()

        Example.configuration(
            plm=self.config['model']['plm'],
            tables=self.output_tables,
            table_path=os.path.join(model_dir, 'tables.json'),
            model_dir=self.model_dir,
            db_dir=os.path.join(model_dir, 'db'))

        self.device = 'cuda' if \
            ('device' not in kwargs or kwargs['device'] == 'gpu') \
            and torch.cuda.is_available() else 'cpu'
        use_device = True if self.device == 'cuda' else False
        self.processor = \
            SubPreprocessor(model_dir=model_dir,
                            db_content=True,
                            use_gpu=use_device)
        self.output_tables = \
            process_tables(self.processor,
                           self.tables)

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
        # use local database
        if data['local_db_path'] is not None and data[
                'local_db_path'] not in self.path_cache:
            self.path_cache.append(data['local_db_path'])
            path = os.path.join(data['local_db_path'], 'tables.json')
            self.tables = json.load(open(path, 'r', encoding='utf-8'))
            self.processor.db_dir = os.path.join(data['local_db_path'], 'db')
            self.output_tables = process_tables(self.processor, self.tables)
            Example.configuration(
                plm=self.config['model']['plm'],
                tables=self.output_tables,
                table_path=path,
                model_dir=self.model_dir,
                db_dir=self.processor.db_dir)

        theresult, sql_label = \
            preprocess_dataset(
                self.processor,
                data,
                self.output_tables,
                data['database_id'],
                self.tables
            )
        output_dataset = process_dataset(self.model_dir, self.processor,
                                         theresult, self.output_tables)
        output_dataset = \
            process_dataset_graph(
                self.graph_processor,
                output_dataset,
                self.output_tables,
                method='lgesql'
            )
        dev_ex = Example(output_dataset[0],
                         self.output_tables[data['database_id']], sql_label)
        current_batch = Batch.from_example_list([dev_ex],
                                                self.device,
                                                train=False)
        return {'batch': current_batch, 'db': data['database_id']}
