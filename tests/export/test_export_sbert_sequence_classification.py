# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest
from collections import OrderedDict

from modelscope.exporters import Exporter, TorchModelExporter
from modelscope.models import Model
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

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_export_sbert_sequence_classification(self):
        model = Model.from_pretrained(self.model_id)
        print(
            Exporter.from_model(model).export_onnx(
                shape=(2, 256), output_dir=self.tmp_dir))
        print(
            TorchModelExporter.from_model(model).export_torch_script(
                shape=(2, 256), output_dir=self.tmp_dir))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_export_outer_module(self):
        from transformers import BertForSequenceClassification, BertTokenizerFast
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased')
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        dummy_inputs = tokenizer(
            tokenizer.unk_token,
            padding='max_length',
            max_length=256,
            return_tensors='pt')
        dynamic_axis = {0: 'batch', 1: 'sequence'}
        inputs = OrderedDict([
            ('input_ids', dynamic_axis),
            ('attention_mask', dynamic_axis),
            ('token_type_ids', dynamic_axis),
        ])
        outputs = OrderedDict({'logits': {0: 'batch'}})
        output_files = TorchModelExporter().export_onnx(
            model=model,
            dummy_inputs=dummy_inputs,
            inputs=inputs,
            outputs=outputs,
            output_dir='/tmp')
        print(output_files)
        output_files = TorchModelExporter().export_torch_script(
            model=model,
            dummy_inputs=dummy_inputs,
            output_dir='/tmp',
            strict=False)
        print(output_files)


if __name__ == '__main__':
    unittest.main()
