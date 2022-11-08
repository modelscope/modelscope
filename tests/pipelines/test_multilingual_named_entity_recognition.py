# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import (LSTMCRFForNamedEntityRecognition,
                                   TransformerCRFForNamedEntityRecognition)
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import (NamedEntityRecognitionThaiPipeline,
                                      NamedEntityRecognitionVietPipeline)
from modelscope.preprocessors import NERPreprocessorThai, NERPreprocessorViet
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class MultilingualNamedEntityRecognitionTest(unittest.TestCase,
                                             DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.named_entity_recognition
        self.model_id = 'damo/nlp_xlmr_named-entity-recognition_thai-ecommerce-title'

    thai_tcrf_model_id = 'damo/nlp_xlmr_named-entity-recognition_thai-ecommerce-title'
    thai_sentence = 'เครื่องชั่งดิจิตอลแบบตั้งพื้น150kg.'

    viet_tcrf_model_id = 'damo/nlp_xlmr_named-entity-recognition_viet-ecommerce-title'
    viet_sentence = 'Nón vành dễ thương cho bé gái'

    multilingual_model_id = 'damo/nlp_raner_named-entity-recognition_multilingual-large-generic'
    ml_stc = 'সমস্ত বেতন নিলামের সাধারণ ব্যবহারিক উদাহরণ বিভিন্ন পেনি নিলাম / বিডিং ফি নিলাম ওয়েবসাইটে পাওয়া যাবে।'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_tcrf_by_direct_model_download_thai(self):
        cache_path = snapshot_download(self.thai_tcrf_model_id)
        tokenizer = NERPreprocessorThai(cache_path)
        model = TransformerCRFForNamedEntityRecognition(
            cache_path, tokenizer=tokenizer)
        pipeline1 = NamedEntityRecognitionThaiPipeline(
            model, preprocessor=tokenizer)
        pipeline2 = pipeline(
            Tasks.named_entity_recognition,
            model=model,
            preprocessor=tokenizer)
        print(f'thai_sentence: {self.thai_sentence}\n'
              f'pipeline1:{pipeline1(input=self.thai_sentence)}')
        print()
        print(f'pipeline2: {pipeline2(input=self.thai_sentence)}')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_tcrf_with_model_from_modelhub_thai(self):
        model = Model.from_pretrained(self.thai_tcrf_model_id)
        tokenizer = NERPreprocessorThai(model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.named_entity_recognition,
            model=model,
            preprocessor=tokenizer)
        print(pipeline_ins(input=self.thai_sentence))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_tcrf_with_model_name_thai(self):
        pipeline_ins = pipeline(
            task=Tasks.named_entity_recognition, model=self.thai_tcrf_model_id)
        print(pipeline_ins(input=self.thai_sentence))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_tcrf_with_model_name_multilingual(self):
        pipeline_ins = pipeline(
            task=Tasks.named_entity_recognition,
            model=self.multilingual_model_id)
        print(pipeline_ins(input=self.ml_stc))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_tcrf_by_direct_model_download_viet(self):
        cache_path = snapshot_download(self.viet_tcrf_model_id)
        tokenizer = NERPreprocessorViet(cache_path)
        model = TransformerCRFForNamedEntityRecognition(
            cache_path, tokenizer=tokenizer)
        pipeline1 = NamedEntityRecognitionVietPipeline(
            model, preprocessor=tokenizer)
        pipeline2 = pipeline(
            Tasks.named_entity_recognition,
            model=model,
            preprocessor=tokenizer)
        print(f'viet_sentence: {self.viet_sentence}\n'
              f'pipeline1:{pipeline1(input=self.viet_sentence)}')
        print()
        print(f'pipeline2: {pipeline2(input=self.viet_sentence)}')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_tcrf_with_model_from_modelhub_viet(self):
        model = Model.from_pretrained(self.viet_tcrf_model_id)
        tokenizer = NERPreprocessorViet(model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.named_entity_recognition,
            model=model,
            preprocessor=tokenizer)
        print(pipeline_ins(input=self.viet_sentence))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_tcrf_with_model_name_viet(self):
        pipeline_ins = pipeline(
            task=Tasks.named_entity_recognition, model=self.viet_tcrf_model_id)
        print(pipeline_ins(input=self.viet_sentence))

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
