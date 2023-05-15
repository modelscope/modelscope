# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class CLIPInterrogatorTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_image_captioning_with_model(self):
        model = Model.from_pretrained('damo/cv_clip-interrogator')
        pipeline_caption = pipeline(
            task=Tasks.image_captioning,
            model=model,
        )
        image = 'data/test/images/image_mplug_vqa.jpg'
        result = pipeline_caption(image)
        print(result[OutputKeys.CAPTION])

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_image_captioning_with_name(self):
        pipeline_caption = pipeline(
            Tasks.image_captioning, model='damo/cv_clip-interrogator')
        image = 'data/test/images/image_mplug_vqa.jpg'
        result = pipeline_caption(image)
        print(result[OutputKeys.CAPTION])


if __name__ == '__main__':
    unittest.main()
