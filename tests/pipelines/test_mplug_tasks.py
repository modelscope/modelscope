# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from PIL import Image

from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class MplugTasksTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_image_captioning_with_model(self):
        model = Model.from_pretrained(
            'damo/mplug_image-captioning_coco_base_en')
        pipeline_caption = pipeline(
            task=Tasks.image_captioning,
            model=model,
        )
        image = Image.open('data/test/images/image_mplug_vqa.jpg')
        result = pipeline_caption({'image': image})
        print(result[OutputKeys.CAPTION])

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_image_captioning_with_name(self):
        pipeline_caption = pipeline(
            Tasks.image_captioning,
            model='damo/mplug_image-captioning_coco_base_en')
        image = Image.open('data/test/images/image_mplug_vqa.jpg')
        result = pipeline_caption({'image': image})
        print(result[OutputKeys.CAPTION])

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_visual_question_answering_with_model(self):
        model = Model.from_pretrained(
            'damo/mplug_visual-question-answering_coco_large_en')
        pipeline_vqa = pipeline(Tasks.visual_question_answering, model=model)
        image = Image.open('data/test/images/image_mplug_vqa.jpg')
        question = 'What is the woman doing?'
        input = {'image': image, 'question': question}
        result = pipeline_vqa(input)
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_visual_question_answering_with_name(self):
        model = 'damo/mplug_visual-question-answering_coco_large_en'
        pipeline_vqa = pipeline(Tasks.visual_question_answering, model=model)
        image = Image.open('data/test/images/image_mplug_vqa.jpg')
        question = 'What is the woman doing?'
        input = {'image': image, 'question': question}
        result = pipeline_vqa(input)
        print(result)


if __name__ == '__main__':
    unittest.main()
