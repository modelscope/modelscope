# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
from typing import Any, Dict

import jieba
import numpy as np
import tensorflow as tf
from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer
from subword_nmt import apply_bpe

from modelscope.metainfo import Pipelines
from modelscope.models.base import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.nlp.translation_pipeline import TranslationPipeline
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_eager_execution()

logger = get_logger()

__all__ = ['InteractiveTranslationPipeline']


@PIPELINES.register_module(
    Tasks.translation, module_name=Pipelines.interactive_translation)
class InteractiveTranslationPipeline(TranslationPipeline):

    def __init__(self, model: Model, **kwargs):
        """Build a interactive translation pipeline with a model dir or a model id in the model hub.

        Args:
            model (`str` or `Model` or module instance): A model instance or a model local dir
                or a model id in the model hub.

        Example:
            >>> from modelscope.pipelines import pipeline
            >>> pipeline_ins = pipeline(task=Tasks.translation,
                model='damo/nlp_imt_translation_zh2en')
            >>> input_sequence = 'Elon Musk, co-founder and chief executive officer of Tesla Motors.'
            >>> input_prefix = "特斯拉汽车公司"
            >>> print(pipeline_ins(input_sequence + "<PREFIX_SPLIT>" + input_prefix))
        """
        super().__init__(model=model, **kwargs)
        model = self.model.model_dir
        tf.reset_default_graph()
        model_path = osp.join(
            osp.join(model, ModelFile.TF_CHECKPOINT_FOLDER), 'ckpt-0')

        self._trg_vocab = dict([
            (w.strip(), i) for i, w in enumerate(open(self._trg_vocab_path))
        ])
        self._len_tgt_vocab = len(self._trg_rvocab)

        self.input_wids = tf.placeholder(
            dtype=tf.int64, shape=[None, None], name='input_wids')

        self.prefix_wids = tf.placeholder(
            dtype=tf.int64, shape=[None, None], name='prefix_wids')

        self.prefix_hit = tf.placeholder(
            dtype=tf.bool, shape=[None, None], name='prefix_hit')

        self.output = {}

        # preprocess
        if self._tgt_lang == 'zh':
            self._tgt_tok = jieba
        else:
            self._tgt_punct_normalizer = MosesPunctNormalizer(
                lang=self._tgt_lang)
            self._tgt_tok = MosesTokenizer(lang=self._tgt_lang)

        # model
        output = self.model(self.input_wids, None, self.prefix_wids,
                            self.prefix_hit)
        self.output.update(output)

        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        self._session = tf.Session(config=tf_config)

        with self._session.as_default() as sess:
            logger.info(f'loading model from {model_path}')
            # load model
            model_loader = tf.train.Saver(tf.global_variables())
            model_loader.restore(sess, model_path)

    def preprocess(self, input: str) -> Dict[str, Any]:
        input_src, prefix = input.split('<PREFIX_SPLIT>', 1)
        if self._src_lang == 'zh':
            input_tok = self._tok.cut(input_src)
            input_tok = ' '.join(list(input_tok))
        else:
            input_src = self._punct_normalizer.normalize(input_src)
            input_tok = self._tok.tokenize(
                input_src, return_str=True, aggressive_dash_splits=True)

        if self._tgt_lang == 'zh':
            prefix = self._tgt_tok.lcut(prefix)
            prefix_tok = ' '.join(list(prefix)[:-1])
        else:
            prefix = self._tgt_punct_normalizer.normalize(prefix)
            prefix = self._tgt_tok.tokenize(
                prefix, return_str=True, aggressive_dash_splits=True).split()
            prefix_tok = ' '.join(prefix[:-1])

        if len(list(prefix)) > 0:
            subword = list(prefix)[-1]
        else:
            subword = ''

        input_bpe = self._bpe.process_line(input_tok)
        prefix_bpe = self._bpe.process_line(prefix_tok)
        input_ids = np.array([[
            self._src_vocab[w]
            if w in self._src_vocab else self.cfg['model']['src_vocab_size']
            for w in input_bpe.strip().split()
        ]])

        prefix_ids = np.array([[
            self._trg_vocab[w]
            if w in self._trg_vocab else self.cfg['model']['trg_vocab_size']
            for w in prefix_bpe.strip().split()
        ]])

        prefix_hit = [[0] * (self._len_tgt_vocab + 1)]

        if subword != '':
            hit_state = False
            for i, w in self._trg_rvocab.items():
                if w.startswith(subword):
                    prefix_hit[0][i] = 1
                    hit_state = True
            if hit_state is False:
                prefix_hit = [[1] * (self._len_tgt_vocab + 1)]
        result = {
            'input_ids': input_ids,
            'prefix_ids': prefix_ids,
            'prefix_hit': np.array(prefix_hit)
        }
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        with self._session.as_default():
            feed_dict = {
                self.input_wids: input['input_ids'],
                self.prefix_wids: input['prefix_ids'],
                self.prefix_hit: input['prefix_hit']
            }
            sess_outputs = self._session.run(self.output, feed_dict=feed_dict)
            return sess_outputs

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        output_seqs = inputs['output_seqs'][0]
        wids = list(output_seqs[0]) + [0]
        wids = wids[:wids.index(0)]
        translation_out = ' '.join([
            self._trg_rvocab[wid] if wid in self._trg_rvocab else '<unk>'
            for wid in wids
        ]).replace('@@ ', '').replace('@@', '')
        translation_out = self._detok.detokenize(translation_out.split())
        result = {OutputKeys.TRANSLATION: translation_out}
        return result
