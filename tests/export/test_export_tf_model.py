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
from modelscope.utils.regress_test_utils import compare_arguments_nested
from modelscope.utils.test_utils import test_level


class TestExportTfModel(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_export_resnet50(self):
        img_path = 'data/test/images/auto_demo.jpg'
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        x_t = tf.convert_to_tensor(x)
        model = ResNet50(weights='imagenet')

        def call_func(inputs):
            return [model.predict(list(inputs.values())[0])]

        output_files = TfModelExporter().export_onnx(
            model=model,
            dummy_inputs={'input': x_t},
            call_func=call_func,
            output_dir=self.tmp_dir)
        print(output_files)


if __name__ == '__main__':
    unittest.main()
