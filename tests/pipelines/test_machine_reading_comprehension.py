# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import ModelForMachineReadingComprehension
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import MachineReadingComprehensionForNERPipeline
from modelscope.preprocessors import \
    MachineReadingComprehensionForNERPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class MachineReadingComprehensionTest(unittest.TestCase):
    sentence = 'Soccer - Japan get lucky win , China in surprise defeat .'
    model_id = 'damo/nlp_roberta_machine-reading-comprehension_for-ner'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_mrc_for_ner_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        tokenizer = MachineReadingComprehensionForNERPreprocessor(cache_path)
        model = ModelForMachineReadingComprehension.from_pretrained(cache_path)
        pipeline1 = MachineReadingComprehensionForNERPipeline(
            model, preprocessor=tokenizer)

        pipeline2 = pipeline(
            Tasks.machine_reading_comprehension,
            model=model,
            preprocessor=tokenizer)
        print(f'sentence: {self.sentence}\n'
              f'pipeline1:{pipeline1(input=self.sentence)}')
        print()
        print(f'pipeline2: {pipeline2(input=self.sentence)}')
        # {'ORG': [], 'PER': [], 'LOC': [' Japan', ' China'], 'MISC': []}

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_mrc_for_ner_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        tokenizer = MachineReadingComprehensionForNERPreprocessor(
            model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.machine_reading_comprehension,
            model=model,
            preprocessor=tokenizer)
        print(f'sentence: {self.sentence}\n'
              f'pipeline:{pipeline_ins(input=self.sentence)}')
        # {'ORG': [], 'PER': [], 'LOC': [' Japan', ' China'], 'MISC': []}

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_mrc_for_ner_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.machine_reading_comprehension, model=self.model_id)
        print(pipeline_ins(input=self.sentence))
        # {'ORG': [], 'PER': [], 'LOC': [' Japan', ' China'], 'MISC': []}


if __name__ == '__main__':
    unittest.main()
