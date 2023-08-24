# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import cv2

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.cv import ImageEditingPipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ImageEditingTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.image_editing
        self.model_id = 'damo/cv_masactrl_image-editing'
        prompts = [
            '',  # source prompt
            'a photo of a running corgi'  # target prompt
        ]
        img = 'https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/public/ModelScope/test/images/corgi.jpg'
        self.input = {'img': img, 'prompts': prompts}
        self.output_image_path = './result.png'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        pipeline = ImageEditingPipeline(cache_path)
        pipeline.group_key = self.task
        edited_img = pipeline(input=self.input)[OutputKeys.OUTPUT_IMG]  # BGR
        cv2.imwrite(self.output_image_path, edited_img)
        print('MasaCtrl pipeline: the edited image path is {}'.format(
            self.output_image_path))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(task=Tasks.image_editing, model=self.model_id)
        edited_img = pipeline_ins(self.input)[OutputKeys.OUTPUT_IMG]  # BGR
        cv2.imwrite(self.output_image_path, edited_img)
        print('MasaCtrl pipeline: the edited image path is {}'.format(
            self.output_image_path))


if __name__ == '__main__':
    unittest.main()
