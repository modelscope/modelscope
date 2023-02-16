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
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_eager_execution()

logger = get_logger()

__all__ = ['TranslationPipeline']


@PIPELINES.register_module(
    Tasks.translation, module_name=Pipelines.csanmt_translation)
class TranslationPipeline(Pipeline):

    def __init__(self, model: Model, **kwargs):
        """Build a translation pipeline with a model dir or a model id in the model hub.

        Args:
            model: A Model instance.
        """
        super().__init__(model=model, **kwargs)
        assert isinstance(self.model, Model), \
            f'please check whether model config exists in {ModelFile.CONFIGURATION}'

        model = self.model.model_dir
        tf.reset_default_graph()

        self.model_path = osp.join(
            osp.join(model, ModelFile.TF_CHECKPOINT_FOLDER), 'ckpt-0')

        self.cfg = Config.from_file(osp.join(model, ModelFile.CONFIGURATION))

        self._src_vocab_path = osp.join(
            model, self.cfg['dataset']['src_vocab']['file'])
        self._src_vocab = dict([
            (w.strip(), i) for i, w in enumerate(open(self._src_vocab_path))
        ])
        self._trg_vocab_path = osp.join(
            model, self.cfg['dataset']['trg_vocab']['file'])
        self._trg_rvocab = dict([
            (i, w.strip()) for i, w in enumerate(open(self._trg_vocab_path))
        ])

        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        self._session = tf.Session(config=tf_config)

        self.input_wids = tf.placeholder(
            dtype=tf.int64, shape=[None, None], name='input_wids')
        self.output = {}

        # preprocess
        self._src_lang = self.cfg['preprocessor']['src_lang']
        self._tgt_lang = self.cfg['preprocessor']['tgt_lang']
        self._src_bpe_path = osp.join(
            model, self.cfg['preprocessor']['src_bpe']['file'])

        if self._src_lang == 'zh':
            self._tok = jieba
        else:
            self._punct_normalizer = MosesPunctNormalizer(lang=self._src_lang)
            self._tok = MosesTokenizer(lang=self._src_lang)
        self._detok = MosesDetokenizer(lang=self._tgt_lang)

        self._bpe = apply_bpe.BPE(open(self._src_bpe_path))

        # model
        output = self.model(self.input_wids)
        self.output.update(output)

        with self._session.as_default() as sess:
            logger.info(f'loading model from {self.model_path}')
            # load model
            self.model_loader = tf.train.Saver(tf.global_variables())
            self.model_loader.restore(sess, self.model_path)

    def preprocess(self, input: str) -> Dict[str, Any]:
        input = input.split('<SENT_SPLIT>')

        if self._src_lang == 'zh':
            input_tok = [self._tok.cut(item) for item in input]
            input_tok = [' '.join(list(item)) for item in input_tok]
        else:
            input = [self._punct_normalizer.normalize(item) for item in input]
            aggressive_dash_splits = True
            if (self._src_lang in ['es', 'fr'] and self._tgt_lang == 'en') or (
                    self._src_lang == 'en' and self._tgt_lang in ['es', 'fr']):
                aggressive_dash_splits = False
            input_tok = [
                self._tok.tokenize(
                    item,
                    return_str=True,
                    aggressive_dash_splits=aggressive_dash_splits)
                for item in input
            ]

        input_bpe = [
            self._bpe.process_line(item).strip().split() for item in input_tok
        ]
        MAX_LENGTH = max([len(item) for item in input_bpe])
        input_ids = np.array([[
            self._src_vocab[w] if w in self._src_vocab else
            self.cfg['model']['src_vocab_size'] - 1 for w in item
        ] + [0] * (MAX_LENGTH - len(item)) for item in input_bpe])
        result = {'input_ids': input_ids}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        with self._session.as_default():
            feed_dict = {self.input_wids: input['input_ids']}
            sess_outputs = self._session.run(self.output, feed_dict=feed_dict)
            return sess_outputs

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        x, y, z = inputs['output_seqs'].shape

        translation_out = []
        for i in range(x):
            output_seqs = inputs['output_seqs'][i]
            wids = list(output_seqs[0]) + [0]
            wids = wids[:wids.index(0)]
            translation = ' '.join([
                self._trg_rvocab[wid] if wid in self._trg_rvocab else '<unk>'
                for wid in wids
            ]).replace('@@ ', '').replace('@@', '')
            translation_out.append(self._detok.detokenize(translation.split()))
        translation_out = '<SENT_SPLIT>'.join(translation_out)
        result = {OutputKeys.TRANSLATION: translation_out}
        return result
