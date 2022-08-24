import os.path as osp
from typing import Any, Dict

import numpy as np
import tensorflow as tf

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

        @param model: A Model instance.
        """
        super().__init__(model=model, **kwargs)
        model = self.model.model_dir
        tf.reset_default_graph()

        model_path = osp.join(
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
            if w in self._src_vocab else self.cfg['model']['src_vocab_size']
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
