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


class TestExportStableDiffusion(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.model_id = 'AI-ModelScope/stable-diffusion-v1-5'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_export_stable_diffusion(self):
        model = Model.from_pretrained(self.model_id)
        Exporter.from_model(model).export_onnx(
            output_path=self.tmp_dir, opset=14)


if __name__ == '__main__':
    unittest.main()
