# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest
from collections import OrderedDict

from modelscope.exporters import Exporter, TorchModelExporter
from modelscope.models import Model
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class TestExportSbertZeroShotClassification(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.model_id = 'damo/nlp_structbert_zero-shot-classification_chinese-base'

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_export_sbert_sequence_classification(self):
        model = Model.from_pretrained(self.model_id)
        print(
            Exporter.from_model(model).export_onnx(
                candidate_labels=[
                    '文化', '体育', '娱乐', '财经', '家居', '汽车', '教育', '科技', '军事'
                ],
                hypothesis_template='这篇文章的标题是{}',
                output_dir=self.tmp_dir))
        print(
            Exporter.from_model(model).export_torch_script(
                candidate_labels=[
                    '文化', '体育', '娱乐', '财经', '家居', '汽车', '教育', '科技', '军事'
                ],
                hypothesis_template='这篇文章的标题是{}',
                output_dir=self.tmp_dir))


if __name__ == '__main__':
    unittest.main()
