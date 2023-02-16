# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict

import torch
from transformers import MT5Tokenizer, XLMRobertaTokenizer

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.config import Config
from modelscope.utils.constant import Fields, ModeKeys, ModelFile
from modelscope.utils.type_assert import type_assert


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.document_grounded_dialog_generate)
class DocumentGroundedDialogGeneratePreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        """The preprocessor for DGDS generate task, based on transformers' tokenizer.

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

        self.top_k = self.config['top_k']
        self.query_sequence_length = self.config['query_sequence_length']
        self.rerank_source_sequence_length = self.config[
            'rerank_source_sequence_length']
        self.source_sequence_length = self.config['source_sequence_length']
        self.target_sequence_length = self.config['target_sequence_length']
        self.rerank_tokenizer = XLMRobertaTokenizer.from_pretrained(
            os.path.join(self.model_dir, 'rerank'))
        self.generation_tokenizer = MT5Tokenizer.from_pretrained(
            os.path.join(self.model_dir, 'generation'))

    @type_assert(object, Dict)
    def __call__(self,
                 data: Dict[str, Any],
                 invoke_mode=ModeKeys.INFERENCE,
                 **preprocessor_param) -> Dict[str, Any]:
        query, context, label = data['query'], data['context'], data.get(
            'label', None)
        query = [
            self.generation_tokenizer.decode(
                self.generation_tokenizer([x],
                                          add_special_tokens=False,
                                          return_tensors='pt')['input_ids'][0]
                [:self.query_sequence_length]) for x in query
        ]

        querys = [x for x in query for i in range(self.top_k)]
        contexts = [x for ctxs in context for x in ctxs[:self.top_k]]
        assert len(querys) == len(contexts)
        rerank_input_ids = self.rerank_tokenizer(
            querys,
            contexts,
            add_special_tokens=True,
            return_tensors='pt',
            max_length=self.rerank_source_sequence_length,
            padding='longest',
            truncation=True)

        generator_inputs = [
            ' '.join([query[i], '<passage>', doc]) for i in range(len(query))
            for doc in context[i][:self.top_k]
        ]
        inputs_tokenizer_outputs = self.generation_tokenizer.batch_encode_plus(
            list(generator_inputs),
            padding=True,
            return_tensors='pt',
            max_length=self.source_sequence_length,
            truncation=True)

        result = {
            'rerank_input_ids': rerank_input_ids,
            'input_ids': inputs_tokenizer_outputs.input_ids,
            'attention_mask': inputs_tokenizer_outputs.attention_mask
        }
        if invoke_mode in (ModeKeys.TRAIN, ModeKeys.EVAL
                           ) and invoke_mode != ModeKeys.INFERENCE:
            result['label_ids'] = self.generation_tokenizer.batch_encode_plus(
                list(label),
                padding=True,
                return_tensors='pt',
                max_length=self.target_sequence_length,
                truncation=True).input_ids

        for k, v in result.items():
            result[k] = v.to(self.device)

        return result
