# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import os
from typing import Any, Dict

import torch
from transformers import XLMRobertaTokenizer

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys, ModelFile
from modelscope.utils.type_assert import type_assert


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.document_grounded_dialog_rerank)
class DocumentGroundedDialogRerankPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, **kwargs):
        """The preprocessor for DGDS rerank task, based on transformers' tokenizer.

        Args:
            model_dir: The model dir containing the essential files to build the tokenizer.
        """
        super().__init__()

        self.model_dir = model_dir
        self.device = 'cuda' \
            if ('device' not in kwargs or kwargs['device'] == 'gpu') and torch.cuda.is_available() \
            else 'cpu'
        self.query_length = kwargs['query_length']
        self.max_seq_length = kwargs['max_seq_length']
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.model_dir)
        if kwargs['tokenizer_resize']:
            special_tokens = [
                '<last_turn>', '<user>', '<agent>', '<response>', '<passage>'
            ]
            self.tokenizer.add_tokens(special_tokens)

    @type_assert(object, Dict)
    def __call__(self, data: Dict[str, Any],
                 **preprocessor_param) -> Dict[str, Any]:
        if 'query' not in data:
            query = data['input']
            passages = data['passages']
            ids = data['id']
            output = data['output']
            positive_pids = data['positive_pids']
            preprocess_output_list = []
            for index in range(len(query)):
                now_query = query[index]
                now_passages = eval(passages[index])
                now_id = ids[index]
                now_output = eval(output[index])
                now_positive_pids = eval(positive_pids[index])
                # query
                query_ids = self.tokenizer(
                    [now_query], add_special_tokens=False,
                    return_tensors='pt')['input_ids'][0][:self.query_length]
                now_query = self.tokenizer.decode(query_ids)
                # passage
                texts_b = []
                for p in now_passages:
                    texts_b.append(' '.join(
                        [now_query, '<passage>', p['text']]))
                passages_input = self.tokenizer(
                    texts_b,
                    add_special_tokens=True,
                    return_tensors='pt',
                    max_length=self.max_seq_length,
                    padding='longest',
                    truncation=True)
                preprocess_output_list.append({
                    'input_ids':
                    passages_input['input_ids'].to(self.device),
                    'attention_mask':
                    passages_input['attention_mask'].to(self.device),
                    'id':
                    now_id,
                    'output':
                    now_output,
                    'positive_pids':
                    now_positive_pids,
                    'passages':
                    now_passages,
                    'query':
                    now_query
                })
            return preprocess_output_list
        else:
            query = data['query']
            passages = data['passages']
            # query
            query_ids = self.tokenizer(
                [query], add_special_tokens=False,
                return_tensors='pt')['input_ids'][0][:self.query_length]
            query = self.tokenizer.decode(query_ids)
            # passage
            texts_b = []
            for p in passages:
                texts_b.append(' '.join([query, '<passage>', p['text']]))
            passages_input = self.tokenizer(
                texts_b,
                add_special_tokens=True,
                return_tensors='pt',
                max_length=self.max_seq_length,
                padding='longest',
                truncation=True)
            result = {n: t.to(self.device) for n, t in passages_input.items()}
        return result
