# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import SpaceForDialogIntent
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import DialogIntentPredictionPipeline
from modelscope.preprocessors import DialogIntentPredictionPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class DialogIntentPredictionTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.task_oriented_conversation
        self.model_id = 'damo/nlp_space_dialog-intent-prediction'

    test_case = [
        'How do I locate my card?',
        'I still have not received my new card, I ordered over a week ago.'
    ]

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        preprocessor = DialogIntentPredictionPreprocessor(model_dir=cache_path)
        model = SpaceForDialogIntent(
            model_dir=cache_path,
            text_field=preprocessor.text_field,
            config=preprocessor.config)

        pipelines = [
            DialogIntentPredictionPipeline(
                model=model, preprocessor=preprocessor),
            pipeline(
                task=Tasks.task_oriented_conversation,
                model=model,
                preprocessor=preprocessor)
        ]

        for my_pipeline, item in list(zip(pipelines, self.test_case)):
            print(my_pipeline(item))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        preprocessor = DialogIntentPredictionPreprocessor(
            model_dir=model.model_dir)

        pipelines = [
            DialogIntentPredictionPipeline(
                model=model, preprocessor=preprocessor),
            pipeline(
                task=Tasks.task_oriented_conversation,
                model=model,
                preprocessor=preprocessor)
        ]

        for my_pipeline, item in list(zip(pipelines, self.test_case)):
            print(my_pipeline(item))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipelines = [pipeline(task=self.task, model=self.model_id)]
        for my_pipeline, item in list(zip(pipelines, self.test_case)):
            print(my_pipeline(item))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
