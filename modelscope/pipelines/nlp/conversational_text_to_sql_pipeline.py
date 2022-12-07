# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Union

import torch
from text2sql_lgesql.utils.example import Example

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.nlp import StarForTextToSql
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import ConversationalTextToSqlPreprocessor
from modelscope.utils.constant import Tasks

__all__ = ['ConversationalTextToSqlPipeline']


@PIPELINES.register_module(
    Tasks.table_question_answering,
    module_name=Pipelines.conversational_text_to_sql)
class ConversationalTextToSqlPipeline(Pipeline):

    def __init__(self,
                 model: Union[StarForTextToSql, str],
                 preprocessor: ConversationalTextToSqlPreprocessor = None,
                 config_file: str = None,
                 device: str = 'gpu',
                 auto_collate=True,
                 **kwargs):
        """use `model` and `preprocessor` to create a conversational text-to-sql prediction pipeline

        Args:
            model (StarForTextToSql): A model instance
            preprocessor (ConversationalTextToSqlPreprocessor): A preprocessor instance
            kwargs (dict, `optional`):
                Extra kwargs passed into the preprocessor's constructor.
        """
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate)
        if preprocessor is None:
            self.preprocessor = ConversationalTextToSqlPreprocessor(
                self.model.model_dir, **kwargs)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): _description_

        Returns:
            Dict[str, str]: the prediction results
        """
        sql = Example.evaluator.obtain_sql(inputs['predict'][0], inputs['db'])
        result = {OutputKeys.OUTPUT: {OutputKeys.TEXT: sql}}
        return result

    def _collate_fn(self, data):
        return data
