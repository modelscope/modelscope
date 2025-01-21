# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest
from collections import OrderedDict

from modelscope.exporters import Exporter
from modelscope.models import Model
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class TestExportOCRRecognition(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.model_id = 'damo/cv_LightweightEdge_ocr-recognitoin-general_damo'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_export_ocr_detection(self):
        model = Model.from_pretrained(
            'damo/cv_LightweightEdge_ocr-recognitoin-general_damo',
            model_revision='v2.4.1')
        Exporter.from_model(model).export_onnx(
            input_shape=(1, 3, 32, 640), output_dir=self.tmp_dir)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_export_ocr_detection_crnn(self):
        model = Model.from_pretrained(
            'damo/cv_crnn_ocr-recognition-general_damo')
        Exporter.from_model(model).export_onnx(
            input_shape=(1, 3, 32, 640), output_dir=self.tmp_dir)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_export_ocr_detection_cvit(self):
        model = Model.from_pretrained(
            'damo/cv_convnextTiny_ocr-recognition-general_damo')
        Exporter.from_model(model).export_onnx(
            input_shape=(3, 3, 32, 300), output_dir=self.tmp_dir)


if __name__ == '__main__':
    unittest.main()
