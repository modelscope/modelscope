# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from PIL import Image

from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class MplugTasksTest(unittest.TestCase, DemoCompatibilityCheck):

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_image_captioning_with_model(self):
        model = Model.from_pretrained(
            'damo/mplug_image-captioning_coco_base_en')
        pipeline_caption = pipeline(
            task=Tasks.image_captioning,
            model=model,
        )
        image = Image.open('data/test/images/image_mplug_vqa.jpg')
        result = pipeline_caption(image)
        print(result[OutputKeys.CAPTION])

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_image_captioning_with_name(self):
        pipeline_caption = pipeline(
            Tasks.image_captioning,
            model='damo/mplug_image-captioning_coco_base_en')
        image = Image.open('data/test/images/image_mplug_vqa.jpg')
        result = pipeline_caption(image)
        print(result[OutputKeys.CAPTION])

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_visual_question_answering_with_model(self):
        model = Model.from_pretrained(
            'damo/mplug_visual-question-answering_coco_large_en')
        pipeline_vqa = pipeline(Tasks.visual_question_answering, model=model)
        image = Image.open('data/test/images/image_mplug_vqa.jpg')
        text = 'What is the woman doing?'
        input = {'image': image, 'text': text}
        result = pipeline_vqa(input)
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_visual_question_answering_with_name(self):
        model = 'damo/mplug_visual-question-answering_coco_large_en'
        pipeline_vqa = pipeline(Tasks.visual_question_answering, model=model)
        image = Image.open('data/test/images/image_mplug_vqa.jpg')
        text = 'What is the woman doing?'
        input = {'image': image, 'text': text}
        result = pipeline_vqa(input)
        print(result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_image_text_retrieval_with_model(self):
        model = Model.from_pretrained(
            'damo/mplug_image-text-retrieval_flickr30k_large_en')
        pipeline_retrieval = pipeline(Tasks.image_text_retrieval, model=model)
        image = Image.open('data/test/images/image-text-retrieval.jpg')
        text = 'Two young guys with shaggy hair look at their hands while hanging out in the yard.'
        input = {'image': image, 'text': text}
        result = pipeline_retrieval(input)
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_image_text_retrieval_with_name(self):
        model = 'damo/mplug_image-text-retrieval_flickr30k_large_en'
        pipeline_retrieval = pipeline(Tasks.image_text_retrieval, model=model)
        image = Image.open('data/test/images/image-text-retrieval.jpg')
        text = 'Two young guys with shaggy hair look at their hands while hanging out in the yard.'
        input = {'image': image, 'text': text}
        result = pipeline_retrieval(input)
        print(result)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_image_captioning_zh_base_with_name(self):
        pipeline_caption = pipeline(
            Tasks.image_captioning,
            model='damo/mplug_image-captioning_coco_base_zh')
        image = Image.open('data/test/images/image_mplug_vqa.jpg')
        result = pipeline_caption(image)
        print(result[OutputKeys.CAPTION])

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_visual_question_answering_zh_base_with_name(self):
        model = 'damo/mplug_visual-question-answering_coco_base_zh'
        pipeline_vqa = pipeline(Tasks.visual_question_answering, model=model)
        image = Image.open('data/test/images/image_mplug_vqa.jpg')
        text = '这个女人在做什么？'
        input = {'image': image, 'text': text}
        result = pipeline_vqa(input)
        print(result)

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
