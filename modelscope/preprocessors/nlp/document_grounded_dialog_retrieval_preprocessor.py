# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict

import torch
from transformers import XLMRobertaTokenizer

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.config import Config
from modelscope.utils.constant import Fields, ModeKeys, ModelFile
from modelscope.utils.type_assert import type_assert


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.document_grounded_dialog_retrieval)
class DocumentGroundedDialogRetrievalPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        """The preprocessor for DGDS retrieval task, based on transformers' tokenizer.

        Args:
            model_dir: The model dir containing the essential files to build the tokenizer.
        """
        super().__init__(*args, **kwargs)

        self.model_dir: str = model_dir
        self.config = Config.from_file(
            os.path.join(self.model_dir, ModelFile.CONFIGURATION))
        self.device = 'cuda' \
            if ('device' not in kwargs or kwargs['device'] == 'gpu') and torch.cuda.is_available() \
            else 'cpu'
        self.query_sequence_length = self.config['query_sequence_length']
        self.context_sequence_length = self.config['context_sequence_length']
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(
            os.path.join(self.model_dir))

    @type_assert(object, Dict)
    def __call__(self,
                 data: Dict[str, Any],
                 invoke_mode=ModeKeys.INFERENCE,
                 input_type='query',
                 **preprocessor_param) -> Dict[str, Any]:
        if invoke_mode in (ModeKeys.TRAIN, ModeKeys.EVAL
                           ) and invoke_mode != ModeKeys.INFERENCE:
            query, positive, negative = data['query'], data['positive'], data[
                'negative']

            query_tokenizer_outputs = self.tokenizer.batch_encode_plus(
                query,
                padding=True,
                return_tensors='pt',
                max_length=self.query_sequence_length,
                truncation=True)

            context_tokenizer_outputs = self.tokenizer.batch_encode_plus(
                positive + negative,
                padding=True,
                return_tensors='pt',
                max_length=self.context_sequence_length,
                truncation=True)

            result = {
                'query_input_ids': query_tokenizer_outputs.input_ids,
                'query_attention_mask': query_tokenizer_outputs.attention_mask,
                'context_input_ids': context_tokenizer_outputs.input_ids,
                'context_attention_mask':
                context_tokenizer_outputs.attention_mask,
                'labels':
                torch.tensor(list(range(len(query))), dtype=torch.long)
            }
        elif input_type == 'query':
            query = data['query']
            query_tokenizer_outputs = self.tokenizer.batch_encode_plus(
                query,
                padding=True,
                return_tensors='pt',
                max_length=self.query_sequence_length,
                truncation=True)
            result = {
                'query_input_ids': query_tokenizer_outputs.input_ids,
                'query_attention_mask': query_tokenizer_outputs.attention_mask,
            }
        else:
            context = data['context']
            context_tokenizer_outputs = self.tokenizer.batch_encode_plus(
                context,
                padding=True,
                return_tensors='pt',
                max_length=self.context_sequence_length,
                truncation=True)
            result = {
                'context_input_ids': context_tokenizer_outputs.input_ids,
                'context_attention_mask':
                context_tokenizer_outputs.attention_mask,
            }

        for k, v in result.items():
            result[k] = v.to(self.device)

        return result
