# Copyright (c) Alibaba, Inc. and its affiliates.
import subprocess
import sys
import tempfile
import unittest

import cv2

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


@unittest.skip('For need realesrgan')
class Text2360PanoramaImageTest(unittest.TestCase):

    def setUp(self) -> None:
        logger.info('start install xformers')
        cmd = [
            sys.executable, '-m', 'pip', 'install', 'xformers', '-f',
            'https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html'
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        logger.info('install xformers finished')

        self.task = Tasks.text_to_360panorama_image
        self.model_id = 'damo/cv_diffusion_text-to-360panorama-image_generation'
        self.prompt = 'The living room'
        self.upscale = False
        self.refinement = False

        self.input = {
            'prompt': self.prompt,
            'upscale': self.upscale,
            'refinement': self.refinement,
        }

    @unittest.skipUnless(test_level() >= 3, 'skip test due to gpu oom')
    def test_run_by_direct_model_download(self):
        from modelscope.pipelines.cv import Text2360PanoramaImagePipeline
        output_image_path = tempfile.NamedTemporaryFile(suffix='.png').name
        cache_path = snapshot_download(self.model_id)
        pipeline = Text2360PanoramaImagePipeline(cache_path)
        pipeline.group_key = self.task
        output = pipeline(inputs=self.input)[OutputKeys.OUTPUT_IMG]
        cv2.imwrite(output_image_path, output)
        print(
            'pipeline: the output image path is {}'.format(output_image_path))

    @unittest.skipUnless(test_level() >= 3, 'skip test due to gpu oom')
    def test_run_with_model_from_modelhub(self):
        from modelscope.pipelines.cv import Text2360PanoramaImagePipeline
        output_image_path = tempfile.NamedTemporaryFile(suffix='.png').name
        pipeline_ins = pipeline(
            task=Tasks.text_to_360panorama_image,
            model=self.model_id,
            model_revision='v1.0.0')
        output = pipeline_ins(inputs=self.input)[OutputKeys.OUTPUT_IMG]
        cv2.imwrite(output_image_path, output)
        print(
            'pipeline: the output image path is {}'.format(output_image_path))


if __name__ == '__main__':
    unittest.main()
