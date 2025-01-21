# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image

from modelscope.exporters import TfModelExporter
from modelscope.models import Model
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import compare_arguments_nested, test_level


class TestExportTfModel(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 2,
                         'test with numpy version == 1.18.1')
    def test_export_csanmt(self):
        model = Model.from_pretrained('damo/nlp_csanmt_translation_en2zh_base')
        print(
            TfModelExporter.from_model(model).export_saved_model(
                output_dir=self.tmp_dir))


if __name__ == '__main__':
    unittest.main()
