# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.multi_modal import MPlugForVisualQuestionAnswering
from modelscope.pipelines import pipeline
from modelscope.pipelines.multi_modal import VisualQuestionAnsweringPipeline
from modelscope.preprocessors import MPlugVisualQuestionAnsweringPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class VisualQuestionAnsweringTest(unittest.TestCase):
    model_id = 'damo/mplug_visual-question-answering_coco_large_en'
    input_vqa = {
        'image': 'data/test/images/image_mplug_vqa.jpg',
        'question': 'What is the woman doing?',
    }

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run(self):
        cache_path = snapshot_download(self.model_id)
        preprocessor = MPlugVisualQuestionAnsweringPreprocessor(cache_path)
        model = MPlugForVisualQuestionAnswering(cache_path)
        pipeline1 = VisualQuestionAnsweringPipeline(
            model, preprocessor=preprocessor)
        pipeline2 = pipeline(
            Tasks.visual_question_answering,
            model=model,
            preprocessor=preprocessor)
        print(f"question: {self.input_vqa['question']}")
        print(f'pipeline1: {pipeline1(self.input_vqa)}')
        print(f'pipeline2: {pipeline2(self.input_vqa)}')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        preprocessor = MPlugVisualQuestionAnsweringPreprocessor(
            model.model_dir)
        pipeline_vqa = pipeline(
            task=Tasks.visual_question_answering,
            model=model,
            preprocessor=preprocessor)
        print(pipeline_vqa(self.input_vqa))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_vqa = pipeline(
            Tasks.visual_question_answering, model=self.model_id)
        print(pipeline_vqa(self.input_vqa))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_vqa = pipeline(task=Tasks.visual_question_answering)
        print(pipeline_vqa(self.input_vqa))


if __name__ == '__main__':
    unittest.main()
