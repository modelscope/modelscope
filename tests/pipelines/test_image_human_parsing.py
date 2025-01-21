# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ImageHumanParsingTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.image_segmentation
        self.model_id_single = 'damo/cv_resnet101_image-single-human-parsing'
        self.model_id_multiple = 'damo/cv_resnet101_image-multiple-human-parsing'

    image_single = 'data/test/images/image_single_human_parsing.jpg'
    image_multiple = 'data/test/images/image_multiple_human_parsing.jpg'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_parsing = pipeline(
            task=Tasks.image_segmentation, model=self.model_id_single)
        print(pipeline_parsing(input=self.image_single)[OutputKeys.LABELS])
        pipeline_parsing = pipeline(
            task=Tasks.image_segmentation, model=self.model_id_multiple)
        print(pipeline_parsing(input=self.image_multiple)[OutputKeys.LABELS])

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id_single)
        pipeline_parsing = pipeline(
            task=Tasks.image_segmentation, model=model, preprocessor=None)
        print(pipeline_parsing(input=self.image_single)[OutputKeys.LABELS])
        model = Model.from_pretrained(self.model_id_multiple)
        pipeline_parsing = pipeline(
            task=Tasks.image_segmentation, model=model, preprocessor=None)
        print(pipeline_parsing(input=self.image_multiple)[OutputKeys.LABELS])


if __name__ == '__main__':
    unittest.main()
