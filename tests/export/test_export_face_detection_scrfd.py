# Copyright (c) Alibaba, Inc. and its affiliates.
'''
For inference using onnx model, please refer to:
https://github.com/deepinsight/insightface/blob/master/detection/scrfd/tools/scrfd.py
'''
import os
import shutil
import tempfile
import unittest
from collections import OrderedDict

from modelscope.exporters import Exporter
from modelscope.models import Model
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class TestExportFaceDetectionSCRFD(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.model_id = 'damo/cv_resnet_facedetection_scrfd10gkps'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_export_face_detection_scrfd(self):
        model = Model.from_pretrained(self.model_id)
        print(Exporter.from_model(model).export_onnx(output_dir=self.tmp_dir))


if __name__ == '__main__':
    unittest.main()
