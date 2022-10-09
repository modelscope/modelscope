# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

from modelscope.exporters import Exporter, TorchModelExporter
from modelscope.models.base import Model
from modelscope.utils.test_utils import test_level


class TestExportSbertSequenceClassification(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.model_id = 'damo/nlp_structbert_sentence-similarity_chinese-base'

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_export_sbert_sequence_classification(self):
        model = Model.from_pretrained(self.model_id)
        print(
            Exporter.from_model(model).export_onnx(
                shape=(2, 256), outputs=self.tmp_dir))
        print(
            TorchModelExporter.from_model(model).export_torch_script(
                shape=(2, 256), outputs=self.tmp_dir))


if __name__ == '__main__':
    unittest.main()
