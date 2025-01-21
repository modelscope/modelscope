# Copyright (c) Alibaba, Inc. and its affiliates.
import subprocess
import sys
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


@unittest.skip('require detectron2-0.3 and torch 1.11.0')
class ImageDefrcnFewShotTest(unittest.TestCase):

    def setUp(self) -> None:
        logger.info('start install detectron2-0.3')
        cmd = [
            sys.executable, '-m', 'pip', 'install', 'detectron2==0.3', '-f',
            'https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html'
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        logger.info('install detectron2-0.3 finished')

        self.task = Tasks.image_fewshot_detection
        self.model_id = 'damo/cv_resnet101_detection_fewshot-defrcn'
        self.image = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_voc2007_000001.jpg'
        self.revision = 'v1.1.0'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id, revision=self.revision)
        pipeline_defrcn = pipeline(
            task=self.task, model=model, model_revision=self.revision)
        print(pipeline_defrcn(input=self.image)[OutputKeys.LABELS])

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_defrcn = pipeline(
            task=self.task, model=self.model_id, model_revision=self.revision)
        print(pipeline_defrcn(input=self.image)[OutputKeys.LABELS])

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_defrcn = pipeline(
            task=self.task, model_revision=self.revision)
        print(pipeline_defrcn(input=self.image)[OutputKeys.LABELS])

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id, revision=self.revision)
        pipeline_defrcn = pipeline(
            self.task, model=cache_path, model_revision=self.revision)
        print(pipeline_defrcn(input=self.image)[OutputKeys.LABELS])


if __name__ == '__main__':
    unittest.main()
