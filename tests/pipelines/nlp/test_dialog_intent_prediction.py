# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from maas_hub.snapshot_download import snapshot_download

from modelscope.models import Model
from modelscope.models.nlp import DialogIntentModel
from modelscope.pipelines import DialogIntentPredictionPipeline, pipeline
from modelscope.preprocessors import DialogIntentPredictionPreprocessor
from modelscope.utils.constant import Tasks


class DialogIntentPredictionTest(unittest.TestCase):
    model_id = 'damo/nlp_space_dialog-intent-prediction'
    test_case = [
        'How do I locate my card?',
        'I still have not received my new card, I ordered over a week ago.'
    ]

    @unittest.skip('test with snapshot_download')
    def test_run(self):
        cache_path = snapshot_download(self.model_id)
        preprocessor = DialogIntentPredictionPreprocessor(model_dir=cache_path)
        model = DialogIntentModel(
            model_dir=cache_path,
            text_field=preprocessor.text_field,
            config=preprocessor.config)

        pipelines = [
            DialogIntentPredictionPipeline(
                model=model, preprocessor=preprocessor),
            pipeline(
                task=Tasks.dialog_intent_prediction,
                model=model,
                preprocessor=preprocessor)
        ]

        for my_pipeline, item in list(zip(pipelines, self.test_case)):
            print(my_pipeline(item))

    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        preprocessor = DialogIntentPredictionPreprocessor(
            model_dir=model.model_dir)

        pipelines = [
            DialogIntentPredictionPipeline(
                model=model, preprocessor=preprocessor),
            pipeline(
                task=Tasks.dialog_intent_prediction,
                model=model,
                preprocessor=preprocessor)
        ]

        for my_pipeline, item in list(zip(pipelines, self.test_case)):
            print(my_pipeline(item))


if __name__ == '__main__':
    unittest.main()
