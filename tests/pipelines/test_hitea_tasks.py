# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class HiTeATasksTest(unittest.TestCase, DemoCompatibilityCheck):

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_video_captioning_with_model(self):
        model = Model.from_pretrained(
            'damo/multi-modal_hitea_video-captioning_base_en')
        pipeline_caption = pipeline(
            task=Tasks.video_captioning,
            model=model,
        )
        video = 'data/test/videos/video_caption_and_qa_test.mp4'
        result = pipeline_caption(video)
        print(result[OutputKeys.CAPTION])

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_video_captioning_with_name(self):
        model = 'damo/multi-modal_hitea_video-captioning_base_en'
        pipeline_caption = pipeline(
            Tasks.video_captioning,
            model=model,
        )
        video = 'data/test/videos/video_caption_and_qa_test.mp4'
        result = pipeline_caption(video)
        print(result[OutputKeys.CAPTION])

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_video_question_answering_with_model(self):
        model = Model.from_pretrained(
            'damo/multi-modal_hitea_video-question-answering_base_en')
        pipeline_vqa = pipeline(Tasks.video_question_answering, model=model)
        video = 'data/test/videos/video_caption_and_qa_test.mp4'
        text = 'How many people are there?'
        input = {'video': video, 'text': text}
        result = pipeline_vqa(input)
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_video_question_answering_with_name(self):
        model = 'damo/multi-modal_hitea_video-question-answering_base_en'
        pipeline_vqa = pipeline(Tasks.video_question_answering, model=model)
        video = 'data/test/videos/video_caption_and_qa_test.mp4'
        text = 'Who teaches a girl how to paint eggs?'
        input = {'video': video, 'text': text}
        result = pipeline_vqa(input)
        print(result)

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
