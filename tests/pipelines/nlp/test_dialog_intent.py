# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from maas_hub.snapshot_download import snapshot_download

from modelscope.models import Model
from modelscope.models.nlp import DialogIntentModel
from modelscope.pipelines import DialogIntentPipeline, pipeline
from modelscope.preprocessors import DialogIntentPreprocessor
from modelscope.utils.constant import Tasks


class DialogGenerationTest(unittest.TestCase):
    model_id = 'damo/nlp_space_dialog-intent'
    test_case = [
        'How do I locate my card?',
        'I still have not received my new card, I ordered over a week ago.'
    ]

    @unittest.skip('test with snapshot_download')
    def test_run(self):
        cache_path = snapshot_download(self.model_id)
        preprocessor = DialogIntentPreprocessor(model_dir=cache_path)
        model = DialogIntentModel(
            model_dir=cache_path,
            text_field=preprocessor.text_field,
            config=preprocessor.config)

        pipelines = [
            DialogIntentPipeline(model=model, preprocessor=preprocessor),
            pipeline(
                task=Tasks.dialog_intent,
                model=model,
                preprocessor=preprocessor)
        ]

        for my_pipeline, item in list(zip(pipelines, self.test_case)):
            print(my_pipeline(item))

    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        preprocessor = DialogIntentPreprocessor(model_dir=model.model_dir)

        pipelines = [
            DialogIntentPipeline(model=model, preprocessor=preprocessor),
            pipeline(
                task=Tasks.dialog_intent,
                model=model,
                preprocessor=preprocessor)
        ]

        for my_pipeline, item in list(zip(pipelines, self.test_case)):
            print(my_pipeline(item))


if __name__ == '__main__':
    unittest.main()
