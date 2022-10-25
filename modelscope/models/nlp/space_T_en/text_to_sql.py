# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Dict, Optional

import torch
from text2sql_lgesql.asdl.asdl import ASDLGrammar
from text2sql_lgesql.asdl.transition_system import TransitionSystem
from text2sql_lgesql.model.model_constructor import Text2SQL

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.base import Tensor
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks

__all__ = ['StarForTextToSql']


@MODELS.register_module(
    Tasks.table_question_answering, module_name=Models.space_T_en)
class StarForTextToSql(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the star model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)
        self.beam_size = 5
        self.config = kwargs.pop(
            'config',
            Config.from_file(
                os.path.join(self.model_dir, ModelFile.CONFIGURATION)))
        self.config.model.model_dir = model_dir
        self.grammar = ASDLGrammar.from_filepath(
            os.path.join(model_dir, 'sql_asdl_v2.txt'))
        self.trans = TransitionSystem.get_class_by_lang('sql')(self.grammar)
        self.arg = self.config.model
        self.device = 'cuda' if \
            ('device' not in kwargs or kwargs['device'] == 'gpu') \
            and torch.cuda.is_available() else 'cpu'
        self.model = Text2SQL(self.arg, self.trans)
        check_point = torch.load(
            open(
                os.path.join(model_dir, ModelFile.TORCH_MODEL_BIN_FILE), 'rb'),
            map_location=self.device)
        self.model.load_state_dict(check_point['model'])

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """return the result by the model

        Args:
            input (Dict[str, Tensor]): the preprocessed data

        Returns:
            Dict[str, Tensor]: results
                Example:

        Example:
            >>> from modelscope.hub.snapshot_download import snapshot_download
            >>> from modelscope.models.nlp import StarForTextToSql
            >>> from modelscope.preprocessors import ConversationalTextToSqlPreprocessor
            >>> test_case = {
                    'database_id': 'employee_hire_evaluation',
                    'local_db_path': None,
                    'utterance': [
                        "I'd like to see Shop names.", 'Which of these are hiring?',
                        'Which shop is hiring the highest number of employees?'
                        ' | do you want the name of the shop ? | Yes'
                    ]
                }
            >>> cache_path = snapshot_download('damo/nlp_star_conversational-text-to-sql')
            >>> preprocessor = ConversationalTextToSqlPreprocessor(
                    model_dir=cache_path,
                    database_id=test_case['database_id'],
                db_content=True)
            >>> model = StarForTextToSql(cache_path, config=preprocessor.config)
            >>> print(model(preprocessor({
                    'utterance': "I'd like to see Shop names.",
                    'history': [],
                    'last_sql': '',
                    'database_id': 'employee_hire_evaluation',
                    'local_db_path': None
                })))
        """
        self.model.eval()
        hyps = self.model.parse(input['batch'], self.beam_size)  #
        db = input['batch'].examples[0].db

        predict = {'predict': hyps, 'db': db}
        return predict
