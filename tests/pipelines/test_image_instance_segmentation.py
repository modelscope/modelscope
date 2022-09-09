# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.cv.image_instance_segmentation import \
    CascadeMaskRCNNSwinModel
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.cv import ImageInstanceSegmentationPipeline
from modelscope.preprocessors import build_preprocessor
from modelscope.utils.config import Config
from modelscope.utils.constant import Fields, ModelFile, Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class ImageInstanceSegmentationTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.image_segmentation
        self.model_id = 'damo/cv_swin-b_image-instance-segmentation_coco'

    image = 'data/test/images/image_instance_segmentation.jpg'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        config_path = os.path.join(model.model_dir, ModelFile.CONFIGURATION)
        cfg = Config.from_file(config_path)
        preprocessor = build_preprocessor(cfg.preprocessor, Fields.cv)
        pipeline_ins = pipeline(
            task=Tasks.image_segmentation,
            model=model,
            preprocessor=preprocessor)
        print(pipeline_ins(input=self.image)[OutputKeys.LABELS])

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.image_segmentation, model=self.model_id)
        print(pipeline_ins(input=self.image)[OutputKeys.LABELS])

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.image_segmentation)
        print(pipeline_ins(input=self.image)[OutputKeys.LABELS])

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        config_path = os.path.join(cache_path, ModelFile.CONFIGURATION)
        cfg = Config.from_file(config_path)
        preprocessor = build_preprocessor(cfg.preprocessor, Fields.cv)
        model = CascadeMaskRCNNSwinModel(cache_path)
        pipeline1 = ImageInstanceSegmentationPipeline(
            model, preprocessor=preprocessor)
        pipeline2 = pipeline(
            Tasks.image_segmentation, model=model, preprocessor=preprocessor)
        print(f'pipeline1:{pipeline1(input=self.image)[OutputKeys.LABELS]}')
        print(f'pipeline2: {pipeline2(input=self.image)[OutputKeys.LABELS]}')

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
