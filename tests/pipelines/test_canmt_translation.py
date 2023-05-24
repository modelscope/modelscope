# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import CanmtForTranslation
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import CanmtTranslationPipeline
from modelscope.preprocessors import CanmtTranslationPreprocessor, Preprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class CanmtTranslationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.competency_aware_translation
        self.model_id = 'damo/nlp_canmt_translation_zh2en_large'

    input = '110例癫痫患者血清抗脑抗体的测定'
    input_2 = '世界是丰富多彩的。'
    input_3 = '行业PE：处于PE估值历史分位较低的行业是房地产、纺织服饰、传媒。'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_direct_download(self):
        cache_path = snapshot_download(self.model_id)
        preprocessor = Preprocessor.from_pretrained(cache_path)
        pipeline1 = CanmtTranslationPipeline(cache_path, preprocessor)
        pipeline2 = pipeline(
            self.task, model=cache_path, preprocessor=preprocessor)
        print(
            f'pipeline1: {pipeline1(self.input)}\npipeline2: {pipeline2(self.input)}'
        )

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_batch(self):
        run_kwargs = {'batch_size': 2}
        pipeline_ins = pipeline(task=self.task, model=self.model_id)
        print(
            'batch: ',
            pipeline_ins([self.input, self.input_2, self.input_3], run_kwargs))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        preprocessor = Preprocessor.from_pretrained(model.model_dir)
        pipeline_ins = pipeline(
            task=self.task, model=model, preprocessor=preprocessor)
        print(pipeline_ins(self.input))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(task=self.task, model=self.model_id)
        print(pipeline_ins(self.input))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=self.task)
        print(pipeline_ins(self.input))


if __name__ == '__main__':
    unittest.main()
