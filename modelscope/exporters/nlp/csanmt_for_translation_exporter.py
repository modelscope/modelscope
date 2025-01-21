import os
from typing import Any, Dict

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import freeze_graph

from modelscope.exporters.builder import EXPORTERS
from modelscope.exporters.tf_model_exporter import TfModelExporter
from modelscope.metainfo import Models
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import compare_arguments_nested

logger = get_logger()

if tf.__version__ >= '2.0':
    tf = tf.compat.v1

tf.logging.set_verbosity(tf.logging.INFO)


@EXPORTERS.register_module(Tasks.translation, module_name=Models.translation)
class CsanmtForTranslationExporter(TfModelExporter):

    def __init__(self, model=None):
        tf.disable_eager_execution()
        super().__init__(model)

        from modelscope.pipelines.nlp.translation_pipeline import \
            TranslationPipeline
        self.pipeline = TranslationPipeline(self.model)

    def generate_dummy_inputs(self, **kwargs) -> Dict[str, Any]:
        return_dict = self.pipeline.preprocess(
            "Alibaba Group's mission is to let the world have no difficult business"
        )
        return {'input_wids': return_dict['input_ids']}

    def export_saved_model(self, output_dir, rtol=None, atol=None, **kwargs):

        def _generate_signature():
            receiver_tensors = {
                'input_wids':
                tf.saved_model.utils.build_tensor_info(
                    self.pipeline.input_wids)
            }
            export_outputs = {
                'output_seqs':
                tf.saved_model.utils.build_tensor_info(
                    self.pipeline.output['output_seqs'])
            }

            signature_def = tf.saved_model.signature_def_utils.build_signature_def(
                receiver_tensors, export_outputs,
                tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

            return {'translation_signature': signature_def}

        with self.pipeline._session.as_default() as sess:
            builder = tf.saved_model.builder.SavedModelBuilder(output_dir)
            builder.add_meta_graph_and_variables(
                sess, [tag_constants.SERVING],
                signature_def_map=_generate_signature(),
                assets_collection=ops.get_collection(
                    ops.GraphKeys.ASSET_FILEPATHS),
                clear_devices=True)
            builder.save()

        dummy_inputs = self.generate_dummy_inputs()
        with tf.Session(graph=tf.Graph()) as sess:
            # Restore model from the saved_model file, that is exported by TensorFlow estimator.
            MetaGraphDef = tf.saved_model.loader.load(sess, ['serve'],
                                                      output_dir)

            # SignatureDef protobuf
            SignatureDef_map = MetaGraphDef.signature_def
            SignatureDef = SignatureDef_map['translation_signature']
            # TensorInfo protobuf
            X_TensorInfo = SignatureDef.inputs['input_wids']
            y_TensorInfo = SignatureDef.outputs['output_seqs']
            X = tf.saved_model.utils.get_tensor_from_tensor_info(
                X_TensorInfo, sess.graph)
            y = tf.saved_model.utils.get_tensor_from_tensor_info(
                y_TensorInfo, sess.graph)
            outputs = sess.run(y, feed_dict={X: dummy_inputs['input_wids']})
            trans_result = self.pipeline.postprocess({'output_seqs': outputs})
            logger.info(trans_result)

        outputs_origin = self.pipeline.forward(
            {'input_ids': dummy_inputs['input_wids']})

        tols = {}
        if rtol is not None:
            tols['rtol'] = rtol
        if atol is not None:
            tols['atol'] = atol
        if not compare_arguments_nested('Output match failed', outputs,
                                        outputs_origin['output_seqs'], **tols):
            raise RuntimeError(
                'Export saved model failed because of validation error.')

        return {'model': output_dir}

    def export_frozen_graph_def(self,
                                output_dir: str,
                                rtol=None,
                                atol=None,
                                **kwargs):
        input_saver_def = self.pipeline.model_loader.as_saver_def()
        inference_graph_def = tf.get_default_graph().as_graph_def()
        for node in inference_graph_def.node:
            node.device = ''

        frozen_dir = os.path.join(output_dir, 'frozen')
        tf.gfile.MkDir(frozen_dir)
        frozen_graph_path = os.path.join(frozen_dir,
                                         'frozen_inference_graph.pb')

        outputs = {
            'output_trans_result':
            tf.identity(
                self.pipeline.output['output_seqs'],
                name='NmtModel/output_trans_result')
        }

        for output_key in outputs:
            tf.add_to_collection('inference_op', outputs[output_key])

        output_node_names = ','.join([
            '%s/%s' % ('NmtModel', output_key)
            for output_key in outputs.keys()
        ])
        print(output_node_names)
        _ = freeze_graph.freeze_graph_with_def_protos(
            input_graph_def=tf.get_default_graph().as_graph_def(),
            input_saver_def=input_saver_def,
            input_checkpoint=self.pipeline.model_path,
            output_node_names=output_node_names,
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph=frozen_graph_path,
            clear_devices=True,
            initializer_nodes='')

        # 5. test frozen.pb
        dummy_inputs = self.generate_dummy_inputs()
        with self.pipeline._session.as_default() as sess:
            sess.run(tf.tables_initializer())

            graph = tf.Graph()
            with tf.gfile.GFile(frozen_graph_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            with graph.as_default():
                tf.import_graph_def(graph_def, name='')
            graph.finalize()

            with tf.Session(graph=graph) as trans_sess:
                outputs = trans_sess.run(
                    'NmtModel/strided_slice_9:0',
                    feed_dict={'input_wids:0': dummy_inputs['input_wids']})
                trans_result = self.pipeline.postprocess(
                    {'output_seqs': outputs})
                logger.info(trans_result)

        outputs_origin = self.pipeline.forward(
            {'input_ids': dummy_inputs['input_wids']})

        tols = {}
        if rtol is not None:
            tols['rtol'] = rtol
        if atol is not None:
            tols['atol'] = atol
        if not compare_arguments_nested('Output match failed', outputs,
                                        outputs_origin['output_seqs'], **tols):
            raise RuntimeError(
                'Export frozen graphdef failed because of validation error.')

        return {'model': frozen_graph_path}

    def export_onnx(self, output_dir: str, opset=13, **kwargs):
        raise NotImplementedError(
            'csanmt model does not support onnx format, consider using saved model instead.'
        )
