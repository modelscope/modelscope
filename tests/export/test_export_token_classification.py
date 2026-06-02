# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest
from collections import OrderedDict

from modelscope.exporters import Exporter, TorchModelExporter
from modelscope.models import Model
from modelscope.utils.constant import Tasks
from modelscope.utils.import_utils import exists
from modelscope.utils.test_utils import test_level


class TestExportTokenClassification(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.model_id = 'damo/nlp_raner_named-entity-recognition_chinese-base-news'

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(
        exists('transformers<5.0'),
        'Skip because transformers version is too high.')
    def test_export_token_classification(self):
        model = Model.from_pretrained(self.model_id)
        with self.subTest(format='onnx'):
            print(
                Exporter.from_model(model).export_onnx(
                    output_dir=self.tmp_dir))
        with self.subTest(format='torchscript'):
            print(
                Exporter.from_model(model).export_torch_script(
                    output_dir=self.tmp_dir))


if __name__ == '__main__':
    unittest.main()
