# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import GPT3ForTextGeneration, PalmForTextGeneration
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import TextGenerationPipeline
from modelscope.preprocessors import TextGenerationPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class TextGenerationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.palm_model_id_zh = 'damo/nlp_palm2.0_text-generation_chinese-base'
        self.palm_model_id_en = 'damo/nlp_palm2.0_text-generation_english-base'
        self.palm_input_zh = """
        本文总结了十个可穿戴产品的设计原则，而这些原则，同样也是笔者认为是这个行业最吸引人的地方：
        1.为人们解决重复性问题；2.从人开始，而不是从机器开始；3.要引起注意，但不要刻意；4.提升用户能力，而不是取代
        """
        self.palm_input_en = """
        The Director of Public Prosecutions who let off Lord Janner over alleged child sex abuse started
        her career at a legal chambers when the disgraced Labour peer was a top QC there . Alison Saunders ,
        54 , sparked outrage last week when she decided the 86-year-old should not face astring of charges
        of paedophilia against nine children because he has dementia . Today , newly-released documents
        revealed damning evidence that abuse was covered up by police andsocial workers for more than 20 years .
        And now it has emerged Mrs Saunders ' law career got off to a flying start when she secured her
        pupillage -- a barrister 's training contract at 1 Garden Court Chambers in London in 1983 .
        """

        self.gpt3_base_model_id = 'damo/nlp_gpt3_text-generation_chinese-base'
        self.gpt3_large_model_id = 'damo/nlp_gpt3_text-generation_chinese-large'
        self.gpt3_input = '我很好奇'

    def run_pipeline_with_model_instance(self, model_id, input):
        model = Model.from_pretrained(model_id)
        preprocessor = TextGenerationPreprocessor(
            model.model_dir,
            model.tokenizer,
            first_sequence='sentence',
            second_sequence=None)
        pipeline_ins = pipeline(
            task=Tasks.text_generation, model=model, preprocessor=preprocessor)
        print(pipeline_ins(input))

    def run_pipeline_with_model_id(self, model_id, input):
        pipeline_ins = pipeline(task=Tasks.text_generation, model=model_id)
        print(pipeline_ins(input))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_palm_zh_with_model_name(self):
        self.run_pipeline_with_model_id(self.palm_model_id_zh,
                                        self.palm_input_zh)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_palm_en_with_model_name(self):
        self.run_pipeline_with_model_id(self.palm_model_id_en,
                                        self.palm_input_en)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_gpt_base_with_model_name(self):
        self.run_pipeline_with_model_id(self.gpt3_base_model_id,
                                        self.gpt3_input)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_gpt_large_with_model_name(self):
        self.run_pipeline_with_model_id(self.gpt3_large_model_id,
                                        self.gpt3_input)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_palm_zh_with_model_instance(self):
        self.run_pipeline_with_model_instance(self.palm_model_id_zh,
                                              self.palm_input_zh)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_palm_en_with_model_instance(self):
        self.run_pipeline_with_model_instance(self.palm_model_id_en,
                                              self.palm_input_en)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_gpt_base_with_model_instance(self):
        self.run_pipeline_with_model_instance(self.gpt3_base_model_id,
                                              self.gpt3_input)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_gpt_large_with_model_instance(self):
        self.run_pipeline_with_model_instance(self.gpt3_large_model_id,
                                              self.gpt3_input)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_palm(self):
        for model_id, input in ((self.palm_model_id_zh, self.palm_input_zh),
                                (self.palm_model_id_en, self.palm_input_en)):
            cache_path = snapshot_download(model_id)
            model = PalmForTextGeneration.from_pretrained(cache_path)
            preprocessor = TextGenerationPreprocessor(
                cache_path,
                model.tokenizer,
                first_sequence='sentence',
                second_sequence=None)
            pipeline1 = TextGenerationPipeline(model, preprocessor)
            pipeline2 = pipeline(
                Tasks.text_generation, model=model, preprocessor=preprocessor)
            print(
                f'pipeline1: {pipeline1(input)}\npipeline2: {pipeline2(input)}'
            )

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_gpt3(self):
        cache_path = snapshot_download(self.gpt3_base_model_id)
        model = GPT3ForTextGeneration(cache_path)
        preprocessor = TextGenerationPreprocessor(
            cache_path,
            model.tokenizer,
            first_sequence='sentence',
            second_sequence=None)
        pipeline1 = TextGenerationPipeline(model, preprocessor)
        pipeline2 = pipeline(
            Tasks.text_generation, model=model, preprocessor=preprocessor)
        print(
            f'pipeline1: {pipeline1(self.gpt3_input)}\npipeline2: {pipeline2(self.gpt3_input)}'
        )

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.text_generation)
        print(pipeline_ins(self.palm_input_zh))


if __name__ == '__main__':
    unittest.main()
