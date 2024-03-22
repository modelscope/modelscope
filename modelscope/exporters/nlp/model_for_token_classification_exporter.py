from collections import OrderedDict
from typing import Any, Dict, Mapping

import torch
from torch import nn

from modelscope.exporters.builder import EXPORTERS
from modelscope.exporters.torch_model_exporter import TorchModelExporter
from modelscope.metainfo import Models
from modelscope.outputs import ModelOutputBase
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.regress_test_utils import (compare_arguments_nested,
                                                 numpify_tensor_nested)


@EXPORTERS.register_module(Tasks.transformer_crf, module_name=Models.tcrf)
@EXPORTERS.register_module(Tasks.token_classification, module_name=Models.tcrf)
@EXPORTERS.register_module(
    Tasks.named_entity_recognition, module_name=Models.tcrf)
@EXPORTERS.register_module(Tasks.part_of_speech, module_name=Models.tcrf)
@EXPORTERS.register_module(Tasks.word_segmentation, module_name=Models.tcrf)
class ModelForSequenceClassificationExporter(TorchModelExporter):

    def generate_dummy_inputs(self, **kwargs) -> Dict[str, Any]:
        """Generate dummy inputs for model exportation to onnx or other formats by tracing.

        Args:
            shape: A tuple of input shape which should have at most two dimensions.
                shape = (1, ) batch_size=1, sequence_length will be taken from the preprocessor.
                shape = (8, 128) batch_size=1, sequence_length=128, which will cover the config of the preprocessor.
            pair(bool, `optional`): Whether to generate sentence pairs or single sentences.

        Returns:
            Dummy inputs.
        """

        assert hasattr(
            self.model, 'model_dir'
        ), 'model_dir attribute is required to build the preprocessor'

        preprocessor = Preprocessor.from_pretrained(
            self.model.model_dir, return_text=False)
        return preprocessor('2023')

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        dynamic_axis = {0: 'batch', 1: 'sequence'}
        return OrderedDict([
            ('input_ids', dynamic_axis),
            ('attention_mask', dynamic_axis),
            ('offset_mapping', dynamic_axis),
            ('label_mask', dynamic_axis),
        ])

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        dynamic_axis = {0: 'batch', 1: 'sequence'}
        return OrderedDict([
            ('predictions', dynamic_axis),
        ])

    def _validate_onnx_model(self,
                             dummy_inputs,
                             model,
                             output,
                             onnx_outputs,
                             rtol: float = None,
                             atol: float = None):
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            logger.warning(
                'Cannot validate the exported onnx file, because '
                'the installation of onnx or onnxruntime cannot be found')
            return
        onnx_model = onnx.load(output)
        onnx.checker.check_model(onnx_model)
        ort_session = ort.InferenceSession(
            output,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        with torch.no_grad():
            model.eval()
            outputs_origin = model.forward(
                *self._decide_input_format(model, dummy_inputs))
        if isinstance(outputs_origin, (Mapping, ModelOutputBase)):
            outputs_origin = list(
                numpify_tensor_nested(outputs_origin).values())
        elif isinstance(outputs_origin, (tuple, list)):
            outputs_origin = list(numpify_tensor_nested(outputs_origin))

        outputs_origin = [outputs_origin[0]
                          ]  # keep `predictions`, drop other outputs

        np_dummy_inputs = numpify_tensor_nested(dummy_inputs)
        np_dummy_inputs['label_mask'] = np_dummy_inputs['label_mask'].astype(
            bool)
        outputs = ort_session.run(onnx_outputs, np_dummy_inputs)
        outputs = numpify_tensor_nested(outputs)
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
