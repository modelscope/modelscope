# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import pickle
import shutil
import tempfile
import unittest

import torch

from modelscope.exporters import Exporter
from modelscope.models import Model
from modelscope.utils.logger import get_logger
from modelscope.utils.regress_test_utils import (compare_arguments_nested,
                                                 numpify_tensor_nested)
from modelscope.utils.test_utils import test_level

INPUT_PKL = 'data/test/audios/input.pkl'

INPUT_NAME = 'input'
OUTPUT_NAME = 'output'

logger = get_logger()


class ExportSpeechSignalProcessTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_export_ans_dfsmn(self):
        model_id = 'damo/speech_dfsmn_ans_psm_48k_causal'
        model = Model.from_pretrained(model_id)
        onnx_info = Exporter.from_model(model).export_onnx(
            output_dir=self.tmp_dir)

        with open(os.path.join(os.getcwd(), INPUT_PKL), 'rb') as f:
            fbank_input = pickle.load(f).cpu()
        self.assertTrue(
            self._validate_onnx_model(fbank_input, model, onnx_info['model']),
            'export onnx failed because of validation error.')

    @staticmethod
    def _validate_onnx_model(dummy_inputs, model, output):
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
        ort_session = ort.InferenceSession(output)
        with torch.no_grad():
            model.eval()
            outputs_origin = model.forward(dummy_inputs)
        outputs_origin = numpify_tensor_nested(outputs_origin)

        input_feed = {INPUT_NAME: dummy_inputs.numpy()}
        outputs = ort_session.run(
            None,
            input_feed,
        )
        outputs = numpify_tensor_nested(outputs[0])

        print(outputs)
        print(outputs_origin)
        return compare_arguments_nested('Onnx model output match failed',
                                        outputs, outputs_origin)


if __name__ == '__main__':
    unittest.main()
