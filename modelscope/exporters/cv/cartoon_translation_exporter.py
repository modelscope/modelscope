import os
from typing import Any, Dict

import tensorflow as tf
from packaging import version

from modelscope.exporters.builder import EXPORTERS
from modelscope.exporters.tf_model_exporter import TfModelExporter
from modelscope.models.cv.cartoon import CartoonModel
from modelscope.utils.logger import get_logger

logger = get_logger(__name__)

if version.parse(tf.__version__) < version.parse('2'):
    pass
else:
    logger.info(
        f'TensorFlow version {_tf_version} found, TF2.x is not supported by CartoonTranslationExporter.'
    )

tf.logging.set_verbosity(tf.logging.INFO)


@EXPORTERS.register_module(module_name=r'cartoon-translation')
class CartoonTranslationExporter(TfModelExporter):

    def __init__(self, model=None):
        super().__init__(model)

    def export_frozen_graph_def(self, ckpt_path: str, frozen_graph_path: str,
                                **kwargs):
        tf.get_variable_scope().reuse_variables()

        input = tf.placeholder(tf.float32, [None, None, 3], name='input_image')
        input = input[:, :, :][tf.newaxis]
        input = input / 127.5 - 1.0

        model = CartoonModel(model_dir='')
        output = model(input)
        final_out = output['output_cartoon'][0]
        final_out = tf.clip_by_value(final_out, -0.999999, 0.999999)
        final_out = (final_out + 1.0) * 127.5
        final_out = tf.cast(final_out, tf.uint8, name='output_image')

        all_vars = tf.trainable_variables()
        gene_vars = [var for var in all_vars if 'generator' in var.name]
        saver = tf.train.Saver(var_list=gene_vars)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:

            sess.run(init)
            saver.restore(sess, ckpt_path)
            frozen_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, output_node_names=['output_image'])
            with open(frozen_graph_path, 'wb') as f:
                f.write(frozen_graph_def.SerializeToString())
            print('freeze done')

        return {'model': frozen_graph_path}

    def export_saved_model(self, output_dir: str, **kwargs):
        raise NotImplementedError(
            'Exporting saved model is not supported by CartoonTranslationExporter currently.'
        )

    def export_onnx(self, output_dir: str, **kwargs):
        raise NotImplementedError(
            'Exporting onnx model is not supported by CartoonTranslationExporter currently.'
        )
