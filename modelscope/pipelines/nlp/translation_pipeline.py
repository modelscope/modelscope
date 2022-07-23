import os.path as osp
from typing import Any, Dict

import numpy as np
import tensorflow as tf

from modelscope.outputs import OutputKeys
from ...hub.snapshot_download import snapshot_download
from ...metainfo import Pipelines
from ...models.nlp import CsanmtForTranslation
from ...utils.constant import ModelFile, Tasks
from ...utils.logger import get_logger
from ..base import Pipeline, Tensor
from ..builder import PIPELINES

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
tf.disable_eager_execution()

logger = get_logger()

__all__ = ['TranslationPipeline']

# constant
PARAMS = {
    'hidden_size': 512,
    'filter_size': 2048,
    'num_heads': 8,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'attention_dropout': 0.0,
    'residual_dropout': 0.0,
    'relu_dropout': 0.0,
    'layer_preproc': 'none',
    'layer_postproc': 'layer_norm',
    'shared_embedding_and_softmax_weights': True,
    'shared_source_target_embedding': True,
    'initializer_scale': 0.1,
    'train_max_len': 100,
    'confidence': 0.9,
    'position_info_type': 'absolute',
    'max_relative_dis': 16,
    'beam_size': 4,
    'lp_rate': 0.6,
    'num_semantic_encoder_layers': 4,
    'max_decoded_trg_len': 100,
    'src_vocab_size': 37006,
    'trg_vocab_size': 37006,
    'vocab_src': 'src_vocab.txt',
    'vocab_trg': 'trg_vocab.txt'
}


@PIPELINES.register_module(
    Tasks.translation, module_name=Pipelines.csanmt_translation)
class TranslationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        super().__init__(model=model)
        model = self.model.model_dir
        tf.reset_default_graph()
        model_path = osp.join(
            osp.join(model, ModelFile.TF_CHECKPOINT_FOLDER), 'ckpt-0')

        self.params = PARAMS
        self._src_vocab_path = osp.join(model, self.params['vocab_src'])
        self._src_vocab = dict([
            (w.strip(), i) for i, w in enumerate(open(self._src_vocab_path))
        ])
        self._trg_vocab_path = osp.join(model, self.params['vocab_trg'])
        self._trg_rvocab = dict([
            (i, w.strip()) for i, w in enumerate(open(self._trg_vocab_path))
        ])

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self._session = tf.Session(config=config)

        self.input_wids = tf.placeholder(
            dtype=tf.int64, shape=[None, None], name='input_wids')
        self.output = {}

        # model
        output = self.model(self.input_wids)
        self.output.update(output)

        with self._session.as_default() as sess:
            logger.info(f'loading model from {model_path}')
            # load model
            model_loader = tf.train.Saver(tf.global_variables())
            model_loader.restore(sess, model_path)

    def preprocess(self, input: str) -> Dict[str, Any]:
        input_ids = np.array([[
            self._src_vocab[w]
            if w in self._src_vocab else self.params['src_vocab_size']
            for w in input.strip().split()
        ]])
        result = {'input_ids': input_ids}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        with self._session.as_default():
            feed_dict = {self.input_wids: input['input_ids']}
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
        result = {OutputKeys.TRANSLATION: translation_out}
        return result
