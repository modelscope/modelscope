# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import PalmForTextGeneration
from modelscope.pipelines import TextGenerationPipeline, pipeline
from modelscope.preprocessors import TextGenerationPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class TextGenerationTest(unittest.TestCase):
    model_id_zh = 'damo/nlp_palm2.0_text-generation_chinese-base'
    model_id_en = 'damo/nlp_palm2.0_text-generation_english-base'
    input_zh = """
    本文总结了十个可穿戴产品的设计原则，而这些原则，同样也是笔者认为是这个行业最吸引人的地方：
    1.为人们解决重复性问题；2.从人开始，而不是从机器开始；3.要引起注意，但不要刻意；4.提升用户能力，而不是取代
    """
    input_en = """
    The Director of Public Prosecutions who let off Lord Janner over alleged child sex abuse started
    her career at a legal chambers when the disgraced Labour peer was a top QC there . Alison Saunders ,
    54 , sparked outrage last week when she decided the 86-year-old should not face astring of charges
    of paedophilia against nine children because he has dementia . Today , newly-released documents
    revealed damning evidence that abuse was covered up by police andsocial workers for more than 20 years .
    And now it has emerged Mrs Saunders ' law career got off to a flying start when she secured her
    pupillage -- a barrister 's training contract at 1 Garden Court Chambers in London in 1983 .
    """

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run(self):
        for model_id, input in ((self.model_id_zh, self.input_zh),
                                (self.model_id_en, self.input_en)):
            cache_path = snapshot_download(model_id)
            model = PalmForTextGeneration(cache_path)
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

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        for model_id, input in ((self.model_id_zh, self.input_zh),
                                (self.model_id_en, self.input_en)):
            model = Model.from_pretrained(model_id)
            preprocessor = TextGenerationPreprocessor(
                model.model_dir,
                model.tokenizer,
                first_sequence='sentence',
                second_sequence=None)
            pipeline_ins = pipeline(
                task=Tasks.text_generation,
                model=model,
                preprocessor=preprocessor)
            print(pipeline_ins(input))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_name(self):
        for model_id, input in ((self.model_id_zh, self.input_zh),
                                (self.model_id_en, self.input_en)):
            pipeline_ins = pipeline(task=Tasks.text_generation, model=model_id)
            print(pipeline_ins(input))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.text_generation)
        print(pipeline_ins(self.input_zh))


if __name__ == '__main__':
    unittest.main()
