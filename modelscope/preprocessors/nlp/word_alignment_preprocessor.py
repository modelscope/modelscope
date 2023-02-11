# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import os.path as osp
from typing import Any, Dict, Optional, Union
import itertools
import numpy as np
import torch

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys
from modelscope.utils.hub import get_model_type
from modelscope.utils.logger import get_logger
from .transformers_tokenizer import NLPTokenizer
from transformers import AutoTokenizer
    
@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.word_alignment)
class WordAlignmentPreprocessor(Preprocessor):
    """The tokenizer preprocessor used in sentence embedding.
    """

    def __init__(self,
                 model_dir: str,
                 sequence_pair='sentence_pair',
                 mode=ModeKeys.INFERENCE,
                 use_fast: bool = False,
                 max_length: int = None,
                 **kwargs):
        """The preprocessor for sentence embedding task, based on transformers' tokenizer.

        Args:
            model_dir: The model dir used to initialize the tokenizer.
            first_sequence: The key of the first sequence.
            second_sequence: The key of the second sequence.
            mode: The mode for the preprocessor.
            use_fast: Use the fast tokenizer or not.
            max_length: The max sequence length which the model supported,
                will be passed into tokenizer as the 'max_length' param.
            **kwargs: Extra args input into the tokenizer's __call__ method.
        """
        self.sequence_pair = sequence_pair

        kwargs[
            'max_length'] = max_length if max_length is not None else kwargs.get(
                'sequence_length', 128)
        self.max_length = kwargs['max_length']
        kwargs.pop('sequence_length', None)
        model_type = None
        
        if model_dir is not None:
            model_type = get_model_type(model_dir)
        self.nlp_tokenizer = NLPTokenizer(
            model_dir, model_type, use_fast=use_fast, tokenize_kwargs=kwargs)
        super().__init__(mode=mode)

    def __call__(self,
                 data: Dict,
                 padding=True,
                 truncation=True,
                 **kwargs) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data Dict:
                keys: the source sentence and the sentences to compare
                values: list of sentences
                Example:
                    {"source_sentence": ["how long it take to get a master's degree"],
                     "sentences_to_compare": ["On average, students take about 18 to 24 months
                     to complete a master's degree.",
                     "On the other hand, some students prefer to go at a slower pace
                     and choose to take several years to complete their studies.",
                     "It can take anywhere from two semesters"]}
        Returns:
            Dict[str, Any]: the preprocessed data
        """
        sentence_pair = data[self.sequence_pair]
        source_sentences, target_sentences = sentence_pair.split('|||')
        # src_lang = data.get("src_lang", 'en_XX')
        # tgt_lang = data.get("tgt_lang", 'en_XX')
        if 'return_tensors' not in kwargs:
            kwargs['return_tensors'] = 'pt' if self.mode == ModeKeys.INFERENCE else None
        
        sent_src, sent_tgt = source_sentences.strip().split(), target_sentences.strip().split()
   
        token_src, token_tgt = [self.nlp_tokenizer.tokenizer.tokenize(word) for word in sent_src], [self.nlp_tokenizer.tokenizer.tokenize(word) for word in sent_tgt]
        wid_src, wid_tgt = [self.nlp_tokenizer.tokenizer.convert_tokens_to_ids(x) for x in token_src], [self.nlp_tokenizer.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
               
        ids_tgt = self.nlp_tokenizer.tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', max_length=self.max_length, prepend_batch_axis=True)['input_ids']               
        ids_src = self.nlp_tokenizer.tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', max_length=self.max_length, prepend_batch_axis=True)['input_ids']
        

        bpe2word_map_src = []
        for i, word_list in enumerate(token_src):
            bpe2word_map_src += [i for x in word_list]
        bpe2word_map_src = torch.Tensor(bpe2word_map_src).type_as(ids_src).view(1, -1)
        bpe2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            bpe2word_map_tgt += [i for x in word_list]
        bpe2word_map_tgt = torch.Tensor(bpe2word_map_tgt).type_as(ids_tgt).view(1, -1) 
        attention_mask_src, attention_mask_tgt = (ids_src!=self.nlp_tokenizer.tokenizer.pad_token_id), (ids_tgt!=self.nlp_tokenizer.tokenizer.pad_token_id)
      
        
        return {'src_input_ids': ids_src,
                'src_attention_mask': attention_mask_src,
                'src_b2w_map': bpe2word_map_src,
                'tgt_input_ids': ids_tgt,
                'tgt_attention_mask': attention_mask_tgt,
                'tgt_b2w_map': bpe2word_map_tgt,
                'threshold': 0.001,
                'bpe_level': False,
                }