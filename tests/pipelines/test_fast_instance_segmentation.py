# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class FastInstanceSegmentationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.image_segmentation
        self.model_id = 'damo/cv_resnet50_fast-instance-segmentation_coco'

    image = 'data/test/images/image_instance_segmentation.jpg'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_parsing = pipeline(
            task=Tasks.image_segmentation, model=self.model_id)
        print(pipeline_parsing(input=self.image)[OutputKeys.LABELS])

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        pipeline_parsing = pipeline(
            task=Tasks.image_segmentation, model=model, preprocessor=None)
        print(pipeline_parsing(input=self.image)[OutputKeys.LABELS])


if __name__ == '__main__':
    unittest.main()
