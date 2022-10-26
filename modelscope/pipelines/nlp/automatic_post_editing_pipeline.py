# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from html import unescape
from typing import Any, Dict

import jieba
import numpy as np
import tensorflow as tf
from sacremoses import (MosesDetokenizer, MosesDetruecaser,
                        MosesPunctNormalizer, MosesTokenizer, MosesTruecaser)
from sentencepiece import SentencePieceProcessor
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops

from modelscope.metainfo import Pipelines
from modelscope.models.base import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.config import Config, ConfigFields
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_eager_execution()

logger = get_logger()

__all__ = ['AutomaticPostEditingPipeline']


@PIPELINES.register_module(
    Tasks.translation, module_name=Pipelines.automatic_post_editing)
class AutomaticPostEditingPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """Build an automatic post editing pipeline with a model dir.

        @param model: Model path for saved pb file
        """
        super().__init__(model=model, **kwargs)
        export_dir = model
        self.cfg = Config.from_file(
            os.path.join(export_dir, ModelFile.CONFIGURATION))
        joint_vocab_file = os.path.join(
            export_dir, self.cfg[ConfigFields.preprocessor]['vocab'])
        self.vocab = dict([(w.strip(), i) for i, w in enumerate(
            open(joint_vocab_file, 'r', encoding='utf8'))])
        self.vocab_reverse = dict([(i, w.strip()) for i, w in enumerate(
            open(joint_vocab_file, 'r', encoding='utf8'))])
        self.unk_id = self.cfg[ConfigFields.preprocessor].get('unk_id', -1)
        strip_unk = self.cfg.get(ConfigFields.postprocessor,
                                 {}).get('strip_unk', True)
        self.unk_token = '' if strip_unk else self.cfg.get(
            ConfigFields.postprocessor, {}).get('unk_token', '<unk>')
        if self.unk_id == -1:
            self.unk_id = len(self.vocab) - 1
        tf.reset_default_graph()
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        self._session = tf.Session(config=tf_config)
        tf.saved_model.loader.load(
            self._session, [tf.python.saved_model.tag_constants.SERVING],
            export_dir)
        default_graph = tf.get_default_graph()
        self.input_src_id_placeholder = default_graph.get_tensor_by_name(
            'Placeholder:0')
        self.input_src_len_placeholder = default_graph.get_tensor_by_name(
            'Placeholder_1:0')
        self.input_mt_id_placeholder = default_graph.get_tensor_by_name(
            'Placeholder_2:0')
        self.input_mt_len_placeholder = default_graph.get_tensor_by_name(
            'Placeholder_3:0')
        output_id_beam = default_graph.get_tensor_by_name(
            'enc2enc/decoder/transpose:0')
        output_len_beam = default_graph.get_tensor_by_name(
            'enc2enc/decoder/Minimum:0')
        output_id = tf.cast(
            tf.map_fn(lambda x: x[0], output_id_beam), dtype=tf.int64)
        output_len = tf.map_fn(lambda x: x[0], output_len_beam)
        self.output = {'output_ids': output_id, 'output_lens': output_len}
        init = tf.global_variables_initializer()
        local_init = tf.local_variables_initializer()
        self._session.run([init, local_init])
        tf.saved_model.loader.load(
            self._session, [tf.python.saved_model.tag_constants.SERVING],
            export_dir)

        # preprocess
        self._src_lang = self.cfg[ConfigFields.preprocessor]['src_lang']
        self._tgt_lang = self.cfg[ConfigFields.preprocessor]['tgt_lang']
        tok_escape = self.cfg[ConfigFields.preprocessor].get(
            'tokenize_escape', False)
        src_tokenizer = MosesTokenizer(lang=self._src_lang)
        mt_tokenizer = MosesTokenizer(lang=self._tgt_lang)
        truecase_model = os.path.join(
            export_dir, self.cfg[ConfigFields.preprocessor]['truecaser'])
        truecaser = MosesTruecaser(load_from=truecase_model)
        sp_model = os.path.join(
            export_dir, self.cfg[ConfigFields.preprocessor]['sentencepiece'])
        sp = SentencePieceProcessor()
        sp.load(sp_model)

        self.src_preprocess = lambda x: ' '.join(
            sp.encode_as_pieces(
                truecaser.truecase(
                    src_tokenizer.tokenize(
                        x, return_str=True, escape=tok_escape),
                    return_str=True)))
        self.mt_preprocess = lambda x: ' '.join(
            sp.encode_as_pieces(
                truecaser.truecase(
                    mt_tokenizer.tokenize(
                        x, return_str=True, escape=tok_escape),
                    return_str=True)))

        # post process, de-bpe, de-truecase, detok
        detruecaser = MosesDetruecaser()
        detokenizer = MosesDetokenizer(lang=self._tgt_lang)
        self.postprocess_fun = lambda x: detokenizer.detokenize(
            detruecaser.detruecase(
                x.replace(' ▁', '@@').replace(' ', '').replace('@@', ' ').
                strip()[1:],
                return_str=True).split())

    def preprocess(self, input: str) -> Dict[str, Any]:
        src, mt = input.split('\005', 1)
        src_sp, mt_sp = self.src_preprocess(src), self.mt_preprocess(mt)
        input_src_ids = np.array(
            [[self.vocab.get(w, self.unk_id) for w in src_sp.strip().split()]])
        input_mt_ids = np.array(
            [[self.vocab.get(w, self.unk_id) for w in mt_sp.strip().split()]])
        input_src_lens = [len(x) for x in input_src_ids]
        input_mt_lens = [len(x) for x in input_mt_ids]
        feed_dict = {
            self.input_src_id_placeholder: input_src_ids,
            self.input_mt_id_placeholder: input_mt_ids,
            self.input_src_len_placeholder: input_src_lens,
            self.input_mt_len_placeholder: input_mt_lens
        }
        return feed_dict

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        with self._session.as_default():
            sess_outputs = self._session.run(self.output, feed_dict=input)
            return sess_outputs

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        output_ids, output_len = inputs['output_ids'][0], inputs[
            'output_lens'][0]
        output_ids = output_ids[:output_len - 1]  # -1 for </s>
        output_tokens = ' '.join([
            self.vocab_reverse.get(wid, self.unk_token) for wid in output_ids
        ])
        post_editing_output = self.postprocess_fun(output_tokens)
        result = {OutputKeys.TRANSLATION: post_editing_output}
        return result
