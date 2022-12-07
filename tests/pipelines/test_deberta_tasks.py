# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import DebertaV2ForMaskedLM
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import FillMaskPipeline
from modelscope.preprocessors import FillMaskTransformersPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class DeBERTaV2TaskTest(unittest.TestCase):
    model_id_deberta = 'damo/nlp_debertav2_fill-mask_chinese-lite'

    ori_text = '你师父差得动你，你师父可差不动我。'
    test_input = '你师父差得动你，你师父可[MASK]不动我。'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        model_dir = snapshot_download(self.model_id_deberta)
        preprocessor = FillMaskTransformersPreprocessor(
            model_dir, first_sequence='sentence', second_sequence=None)
        model = DebertaV2ForMaskedLM.from_pretrained(model_dir)
        pipeline1 = FillMaskPipeline(model, preprocessor)
        pipeline2 = pipeline(
            Tasks.fill_mask, model=model, preprocessor=preprocessor)
        ori_text = self.ori_text
        test_input = self.test_input
        print(f'\nori_text: {ori_text}\ninput: {test_input}\npipeline1: '
              f'{pipeline1(test_input)}\npipeline2: {pipeline2(test_input)}\n')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        # sbert
        print(self.model_id_deberta)
        model = Model.from_pretrained(self.model_id_deberta)
        preprocessor = FillMaskTransformersPreprocessor(
            model.model_dir, first_sequence='sentence', second_sequence=None)
        pipeline_ins = pipeline(
            task=Tasks.fill_mask, model=model, preprocessor=preprocessor)
        print(
            f'\nori_text: {self.ori_text}\ninput: {self.test_input}\npipeline: '
            f'{pipeline_ins(self.test_input)}\n')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.fill_mask, model=self.model_id_deberta)
        ori_text = self.ori_text
        test_input = self.test_input
        print(f'\nori_text: {ori_text}\ninput: {test_input}\npipeline: '
              f'{pipeline_ins(test_input)}\n')


if __name__ == '__main__':
    unittest.main()
