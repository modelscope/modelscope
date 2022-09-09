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
from modelscope.preprocessors.star.fields import (SubPreprocessor,
                                                  process_tables)
from modelscope.utils.constant import Tasks

__all__ = ['ConversationalTextToSqlPipeline']


@PIPELINES.register_module(
    Tasks.conversational_text_to_sql,
    module_name=Pipelines.conversational_text_to_sql)
class ConversationalTextToSqlPipeline(Pipeline):

    def __init__(self,
                 model: Union[StarForTextToSql, str],
                 preprocessor: ConversationalTextToSqlPreprocessor = None,
                 **kwargs):
        """use `model` and `preprocessor` to create a conversational text-to-sql prediction pipeline

        Args:
            model (StarForTextToSql): a model instance
            preprocessor (ConversationalTextToSqlPreprocessor):
                a preprocessor instance
        """
        model = model if isinstance(
            model, StarForTextToSql) else Model.from_pretrained(model)
        if preprocessor is None:
            preprocessor = ConversationalTextToSqlPreprocessor(model.model_dir)

        preprocessor.device = 'cuda' if \
            ('device' not in kwargs or kwargs['device'] == 'gpu') \
            and torch.cuda.is_available() else 'cpu'
        use_device = True if preprocessor.device == 'cuda' else False
        preprocessor.processor = \
            SubPreprocessor(model_dir=model.model_dir,
                            db_content=True,
                            use_gpu=use_device)
        preprocessor.output_tables = \
            process_tables(preprocessor.processor,
                           preprocessor.tables)
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): _description_

        Returns:
            Dict[str, str]: the prediction results
        """
        sql = Example.evaluator.obtain_sql(inputs['predict'][0], inputs['db'])
        result = {OutputKeys.TEXT: sql}
        return result

    def _collate_fn(self, data):
        return data
