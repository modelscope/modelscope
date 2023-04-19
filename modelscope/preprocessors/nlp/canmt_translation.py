# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
from typing import Any, Dict

import jieba
import torch
from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer
from subword_nmt import apply_bpe

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.config import Config
from modelscope.utils.constant import Fields, ModelFile
from .text_clean import TextClean


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.canmt_translation)
class CanmtTranslationPreprocessor(Preprocessor):
    """The preprocessor used in text correction task.
    """

    def __init__(self,
                 model_dir: str,
                 max_length: int = None,
                 *args,
                 **kwargs):
        from fairseq.data import Dictionary
        """preprocess the data via the vocab file from the `model_dir` path

        Args:
            model_dir (str): model path
        """
        super().__init__(*args, **kwargs)
        self.cfg = Config.from_file(
            osp.join(model_dir, ModelFile.CONFIGURATION))
        self.vocab_src = Dictionary.load(osp.join(model_dir, 'dict.src.txt'))
        self.vocab_tgt = Dictionary.load(osp.join(model_dir, 'dict.tgt.txt'))
        self.padding_value = self.vocab_src.pad()
        self.max_length = max_length + 1 if max_length is not None else 129  # 1 is eos token

        self.src_lang = self.cfg['preprocessor']['src_lang']
        self.tgt_lang = self.cfg['preprocessor']['tgt_lang']
        self.tc = TextClean()

        if self.src_lang == 'zh':
            self.tok = jieba
        else:
            self.punct_normalizer = MosesPunctNormalizer(lang=self.src_lang)
            self.tok = MosesTokenizer(lang=self.src_lang)

        self.src_bpe_path = osp.join(
            model_dir, self.cfg['preprocessor']['src_bpe']['file'])
        self.bpe = apply_bpe.BPE(open(self.src_bpe_path))

    def __call__(self, input: str) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data (str): a sentence
                Example:
                    '随着中国经济突飞猛近，建造工业与日俱增'
        Returns:
            Dict[str, Any]: the preprocessed data
            Example:
            {'net_input':
                {'src_tokens':tensor([1,2,3,4]),
                'src_lengths': tensor([4])}
            }
        """
        if self.src_lang == 'zh':
            input = self.tc.clean(input)
            input_tok = self.tok.cut(input)
            input_tok = ' '.join(list(input_tok))
        else:
            input = [self._punct_normalizer.normalize(item) for item in input]
            input_tok = [
                self.tok.tokenize(
                    item, return_str=True, aggressive_dash_splits=True)
                for item in input
            ]

        input_bpe = self.bpe.process_line(input_tok).strip().split()
        text = ' '.join([x for x in input_bpe])

        inputs = self.vocab_src.encode_line(
            text, append_eos=True, add_if_not_exist=False)
        prev_inputs = torch.roll(inputs, shifts=1)
        lengths = inputs.size()[0]
        max_len = min(self.max_length, lengths)

        padding = torch.tensor(
            [self.padding_value] *  # noqa: W504
            (max_len - lengths),
            dtype=inputs.dtype)
        sources = torch.unsqueeze(torch.cat([inputs, padding]), dim=0)
        inputs = torch.unsqueeze(torch.cat([padding, inputs]), dim=0)
        prev_inputs = torch.unsqueeze(torch.cat([prev_inputs, padding]), dim=0)
        lengths = torch.tensor([lengths])
        out = {
            'src_tokens': inputs,
            'src_lengths': lengths,
            'prev_src_tokens': prev_inputs,
            'sources': sources
        }

        return out
