import os.path as osp
from threading import Lock
from typing import Any, Dict

import numpy as np
import tensorflow as tf

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.config import Config
from modelscope.utils.constant import Frameworks, ModelFile, Tasks
from modelscope.utils.logger import get_logger

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_eager_execution()

logger = get_logger()

__all__ = ['TranslationPipeline']


@PIPELINES.register_module(
    Tasks.translation, module_name=Pipelines.csanmt_translation)
class TranslationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        tf.reset_default_graph()
        self.framework = Frameworks.tf
        self.device_name = 'cpu'

        super().__init__(model=model)

        model_path = osp.join(
            osp.join(model, ModelFile.TF_CHECKPOINT_FOLDER), 'ckpt-0')

        self.cfg = Config.from_file(osp.join(model, ModelFile.CONFIGURATION))

        self.params = {}
        self._override_params_from_file()

        self._src_vocab_path = osp.join(model, self.params['vocab_src'])
        self._src_vocab = dict([
            (w.strip(), i) for i, w in enumerate(open(self._src_vocab_path))
        ])
        self._trg_vocab_path = osp.join(model, self.params['vocab_trg'])
        self._trg_rvocab = dict([
            (i, w.strip()) for i, w in enumerate(open(self._trg_vocab_path))
        ])

        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        self._session = tf.Session(config=tf_config)

        self.input_wids = tf.placeholder(
            dtype=tf.int64, shape=[None, None], name='input_wids')
        self.output = {}

        # model
        self.model = CsanmtForTranslation(model_path, params=self.params)
        output = self.model(self.input_wids)
        self.output.update(output)

        with self._session.as_default() as sess:
            logger.info(f'loading model from {model_path}')
            # load model
            model_loader = tf.train.Saver(tf.global_variables())
            model_loader.restore(sess, model_path)

    def _override_params_from_file(self):

        # model
        self.params['hidden_size'] = self.cfg['model']['hidden_size']
        self.params['filter_size'] = self.cfg['model']['filter_size']
        self.params['num_heads'] = self.cfg['model']['num_heads']
        self.params['num_encoder_layers'] = self.cfg['model'][
            'num_encoder_layers']
        self.params['num_decoder_layers'] = self.cfg['model'][
            'num_decoder_layers']
        self.params['layer_preproc'] = self.cfg['model']['layer_preproc']
        self.params['layer_postproc'] = self.cfg['model']['layer_postproc']
        self.params['shared_embedding_and_softmax_weights'] = self.cfg[
            'model']['shared_embedding_and_softmax_weights']
        self.params['shared_source_target_embedding'] = self.cfg['model'][
            'shared_source_target_embedding']
        self.params['initializer_scale'] = self.cfg['model'][
            'initializer_scale']
        self.params['position_info_type'] = self.cfg['model'][
            'position_info_type']
        self.params['max_relative_dis'] = self.cfg['model']['max_relative_dis']
        self.params['num_semantic_encoder_layers'] = self.cfg['model'][
            'num_semantic_encoder_layers']
        self.params['src_vocab_size'] = self.cfg['model']['src_vocab_size']
        self.params['trg_vocab_size'] = self.cfg['model']['trg_vocab_size']
        self.params['attention_dropout'] = 0.0
        self.params['residual_dropout'] = 0.0
        self.params['relu_dropout'] = 0.0

        # dataset
        self.params['vocab_src'] = self.cfg['dataset']['src_vocab']['file']
        self.params['vocab_trg'] = self.cfg['dataset']['trg_vocab']['file']

        # train
        self.params['train_max_len'] = self.cfg['train']['train_max_len']
        self.params['confidence'] = self.cfg['train']['confidence']

        # evaluation
        self.params['beam_size'] = self.cfg['evaluation']['beam_size']
        self.params['lp_rate'] = self.cfg['evaluation']['lp_rate']
        self.params['max_decoded_trg_len'] = self.cfg['evaluation'][
            'max_decoded_trg_len']

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
