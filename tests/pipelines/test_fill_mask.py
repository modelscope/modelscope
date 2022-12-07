# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from regex import R

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import SbertForMaskedLM, VecoForMaskedLM
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import FillMaskPipeline
from modelscope.preprocessors import FillMaskTransformersPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.regress_test_utils import IgnoreKeyFn, MsRegressTool
from modelscope.utils.test_utils import test_level


class FillMaskTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.fill_mask
        self.model_id = 'damo/nlp_veco_fill-mask-large'

    model_id_sbert = {
        'zh': 'damo/nlp_structbert_fill-mask_chinese-large',
        'en': 'damo/nlp_structbert_fill-mask_english-large'
    }
    model_id_veco = 'damo/nlp_veco_fill-mask-large'
    model_id_bert = 'damo/nlp_bert_fill-mask_chinese-base'

    ori_texts = {
        'zh':
        '段誉轻挥折扇，摇了摇头，说道：“你师父是你的师父，你师父可不是我的师父。'
        '你师父差得动你，你师父可差不动我。',
        'en':
        'Everything in what you call reality is really just a reflection of your '
        'consciousness. Your whole universe is just a mirror reflection of your story.'
    }

    test_inputs = {
        'zh':
        '段誉轻[MASK]折扇，摇了摇[MASK]，[MASK]道：“你师父是你的[MASK][MASK]，你'
        '师父可不是[MASK]的师父。你师父差得动你，你师父可[MASK]不动我。',
        'en':
        'Everything in [MASK] you call reality is really [MASK] a reflection of your '
        '[MASK]. Your [MASK] universe is just a mirror [MASK] of your story.'
    }
    regress_tool = MsRegressTool(baseline=False)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        # sbert
        for language in ['zh']:
            model_dir = snapshot_download(self.model_id_sbert[language])
            preprocessor = FillMaskTransformersPreprocessor(
                model_dir, first_sequence='sentence', second_sequence=None)
            model = SbertForMaskedLM.from_pretrained(model_dir)
            pipeline1 = FillMaskPipeline(model, preprocessor)
            pipeline2 = pipeline(
                Tasks.fill_mask, model=model, preprocessor=preprocessor)
            ori_text = self.ori_texts[language]
            test_input = self.test_inputs[language]
            print(
                f'\nori_text: {ori_text}\ninput: {test_input}\npipeline1: '
                f'{pipeline1(test_input)}\npipeline2: {pipeline2(test_input)}\n'
            )

        # veco
        model_dir = snapshot_download(self.model_id_veco)
        preprocessor = FillMaskTransformersPreprocessor(
            model_dir, first_sequence='sentence', second_sequence=None)
        model = VecoForMaskedLM.from_pretrained(model_dir)
        pipeline1 = FillMaskPipeline(model, preprocessor)
        pipeline2 = pipeline(
            Tasks.fill_mask, model=model, preprocessor=preprocessor)
        for language in ['zh', 'en']:
            ori_text = self.ori_texts[language]
            test_input = self.test_inputs[language].replace('[MASK]', '<mask>')
            print(
                f'\nori_text: {ori_text}\ninput: {test_input}\npipeline1: '
                f'{pipeline1(test_input)}\npipeline2: {pipeline2(test_input)}\n'
            )

        # bert
        language = 'zh'
        model_dir = snapshot_download(self.model_id_bert)
        preprocessor = FillMaskTransformersPreprocessor(
            model_dir, first_sequence='sentence', second_sequence=None)
        model = Model.from_pretrained(model_dir)
        pipeline1 = FillMaskPipeline(model, preprocessor)
        pipeline2 = pipeline(
            Tasks.fill_mask, model=model, preprocessor=preprocessor)
        ori_text = self.ori_texts[language]
        test_input = self.test_inputs[language]
        print(f'\nori_text: {ori_text}\ninput: {test_input}\npipeline1: '
              f'{pipeline1(test_input)}\npipeline2: {pipeline2(test_input)}\n')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):

        # sbert
        for language in ['zh']:
            print(self.model_id_sbert[language])
            model = Model.from_pretrained(self.model_id_sbert[language])
            preprocessor = FillMaskTransformersPreprocessor(
                model.model_dir,
                first_sequence='sentence',
                second_sequence=None)
            pipeline_ins = pipeline(
                task=Tasks.fill_mask, model=model, preprocessor=preprocessor)
            with self.regress_tool.monitor_module_single_forward(
                    pipeline_ins.model,
                    f'fill_mask_sbert_{language}',
                    compare_fn=IgnoreKeyFn('.*intermediate_act_fn')):
                print(
                    f'\nori_text: {self.ori_texts[language]}\ninput: {self.test_inputs[language]}\npipeline: '
                    f'{pipeline_ins(self.test_inputs[language])}\n')

        # veco
        model = Model.from_pretrained(self.model_id_veco)
        preprocessor = FillMaskTransformersPreprocessor(
            model.model_dir, first_sequence='sentence', second_sequence=None)
        pipeline_ins = pipeline(
            Tasks.fill_mask, model=model, preprocessor=preprocessor)
        for language in ['zh', 'en']:
            ori_text = self.ori_texts[language]
            test_input = self.test_inputs[language].replace('[MASK]', '<mask>')
            with self.regress_tool.monitor_module_single_forward(
                    pipeline_ins.model,
                    f'fill_mask_veco_{language}',
                    compare_fn=IgnoreKeyFn('.*intermediate_act_fn')):
                print(
                    f'\nori_text: {ori_text}\ninput: {test_input}\npipeline: '
                    f'{pipeline_ins(test_input)}\n')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        # veco
        pipeline_ins = pipeline(task=Tasks.fill_mask, model=self.model_id_veco)
        for language in ['zh', 'en']:
            ori_text = self.ori_texts[language]
            test_input = self.test_inputs[language].replace('[MASK]', '<mask>')
            print(f'\nori_text: {ori_text}\ninput: {test_input}\npipeline: '
                  f'{pipeline_ins(test_input)}\n')

        # structBert
        language = 'zh'
        pipeline_ins = pipeline(
            task=Tasks.fill_mask, model=self.model_id_sbert[language])
        print(
            f'\nori_text: {self.ori_texts[language]}\ninput: {self.test_inputs[language]}\npipeline: '
            f'{pipeline_ins(self.test_inputs[language])}\n')

        # Bert
        language = 'zh'
        pipeline_ins = pipeline(task=Tasks.fill_mask, model=self.model_id_bert)
        print(
            f'\nori_text: {self.ori_texts[language]}\ninput: {self.test_inputs[language]}\npipeline: '
            f'{pipeline_ins(self.test_inputs[language])}\n')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.fill_mask)
        language = 'en'
        ori_text = self.ori_texts[language]
        test_input = self.test_inputs[language].replace('[MASK]', '<mask>')
        print(f'\nori_text: {ori_text}\ninput: {test_input}\npipeline: '
              f'{pipeline_ins(test_input)}\n')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
