# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.cv.image_instance_segmentation import MaskDINOSwinModel
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.cv import MaskDINOInstanceSegmentationPipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class MaskDINOInstanceSegmentationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.image_segmentation
        self.model_id = 'damo/cv_maskdino-swin-l_image-instance-segmentation_coco'

    image = 'data/test/images/image_instance_segmentation.jpg'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.image_segmentation, model=self.model_id)
        print(pipeline_ins(input=self.image)[OutputKeys.LABELS])

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        pipeline_ins = pipeline(
            task=Tasks.image_segmentation, model=model, preprocessor=None)
        print(pipeline_ins(input=self.image)[OutputKeys.LABELS])

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        model = MaskDINOSwinModel(cache_path)
        pipeline1 = MaskDINOInstanceSegmentationPipeline(
            model, preprocessor=None)
        pipeline2 = pipeline(
            Tasks.image_segmentation, model=model, preprocessor=None)
        print(f'pipeline1:{pipeline1(input=self.image)[OutputKeys.LABELS]}')
        print(f'pipeline2: {pipeline2(input=self.image)[OutputKeys.LABELS]}')


if __name__ == '__main__':
    unittest.main()
