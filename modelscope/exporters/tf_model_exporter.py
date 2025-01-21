# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Callable, Dict, Mapping

import tensorflow as tf

from modelscope.outputs import ModelOutputBase
from modelscope.utils.constant import ModelFile
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import compare_arguments_nested
from .base import Exporter

logger = get_logger()


class TfModelExporter(Exporter):

    def generate_dummy_inputs(self, **kwargs) -> Dict[str, Any]:
        """Generate dummy inputs for model exportation to onnx or other formats by tracing.

        Returns:
            Dummy inputs that matches the specific model input, the matched preprocessor can be used here.
        """
        return None

    def export_onnx(self, output_dir: str, opset=13, **kwargs):
        model = self.model if 'model' not in kwargs else kwargs.pop('model')
        onnx_file = os.path.join(output_dir, ModelFile.ONNX_MODEL_FILE)
        self._tf2_export_onnx(model, onnx_file, opset=opset, **kwargs)
        return {'model': onnx_file}

    def export_saved_model(self, output_dir: str, **kwargs):
        raise NotImplementedError()

    def export_frozen_graph_def(self, output_dir: str, **kwargs):
        raise NotImplementedError()

    def _tf2_export_onnx(self,
                         model,
                         output: str,
                         opset: int = 13,
                         validation: bool = True,
                         rtol: float = None,
                         atol: float = None,
                         call_func: Callable = None,
                         **kwargs):
        logger.info(
            'Important: This exporting function only supports models of tf2.0 or above.'
        )
        import onnx
        import tf2onnx
        dummy_inputs = self.generate_dummy_inputs(
            **kwargs) if 'dummy_inputs' not in kwargs else kwargs.pop(
                'dummy_inputs')
        if dummy_inputs is None:
            raise NotImplementedError(
                'Model property dummy_inputs,inputs,outputs must be set.')

        input_signature = [
            tf.TensorSpec.from_tensor(tensor, name=key)
            for key, tensor in dummy_inputs.items()
        ]
        onnx_model, _ = tf2onnx.convert.from_keras(
            model, input_signature, opset=opset)
        onnx.save(onnx_model, output)

        if validation:
            self._validate_model(dummy_inputs, model, output, rtol, atol,
                                 call_func)

    def _validate_model(
        self,
        dummy_inputs,
        model,
        output,
        rtol: float = None,
        atol: float = None,
        call_func: Callable = None,
    ):
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            logger.warn(
                'Cannot validate the exported onnx file, because '
                'the installation of onnx or onnxruntime cannot be found')
            return

        def tensor_nested_numpify(tensors):
            if isinstance(tensors, (list, tuple)):
                return type(tensors)(tensor_nested_numpify(t) for t in tensors)
            if isinstance(tensors, Mapping):
                # return dict
                return {
                    k: tensor_nested_numpify(t)
                    for k, t in tensors.items()
                }
            if isinstance(tensors, tf.Tensor):
                t = tensors.cpu()
                return t.numpy()
            return tensors

        onnx_model = onnx.load(output)
        onnx.checker.check_model(onnx_model, full_check=True)
        ort_session = ort.InferenceSession(
            output,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        outputs_origin = call_func(
            dummy_inputs) if call_func is not None else model(dummy_inputs)
        if isinstance(outputs_origin, (Mapping, ModelOutputBase)):
            outputs_origin = list(
                tensor_nested_numpify(outputs_origin).values())
        elif isinstance(outputs_origin, (tuple, list)):
            outputs_origin = list(tensor_nested_numpify(outputs_origin))
        outputs = ort_session.run(
            None,
            tensor_nested_numpify(dummy_inputs),
        )
        outputs = tensor_nested_numpify(outputs)
        if isinstance(outputs, dict):
            outputs = list(outputs.values())
        elif isinstance(outputs, tuple):
            outputs = list(outputs)

        tols = {}
        if rtol is not None:
            tols['rtol'] = rtol
        if atol is not None:
            tols['atol'] = atol
        if not compare_arguments_nested('Onnx model output match failed',
                                        outputs, outputs_origin, **tols):
            raise RuntimeError(
                'export onnx failed because of validation error.')
