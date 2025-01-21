# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import cv2

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class CartoonStableDiffusionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.text_to_image_synthesis
        self.model_id = 'damo/cv_cartoon_stable_diffusion_design'
        self.model_id_illu = 'damo/cv_cartoon_stable_diffusion_illustration'
        self.model_id_watercolor = 'damo/cv_cartoon_stable_diffusion_watercolor'
        self.model_id_flat = 'damo/cv_cartoon_stable_diffusion_flat'
        self.model_id_clipart = 'damo/cv_cartoon_stable_diffusion_clipart'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_default(self):
        pipe = pipeline(
            task=self.task, model=self.model_id, model_revision='v1.0.0')
        output = pipe(
            {'text': 'sks style, a portrait painting of Johnny Depp'})
        cv2.imwrite('result_design.png', output['output_imgs'][0])
        print('Image saved to result_design.png')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_illustration(self):
        pipe = pipeline(
            task=self.task, model=self.model_id_illu, model_revision='v1.0.0')
        output = pipe(
            {'text': 'sks style, a portrait painting of Johnny Depp'})
        cv2.imwrite('result_illu.png', output['output_imgs'][0])
        print('Image saved to result_illu.png')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_watercolor(self):
        pipe = pipeline(
            task=self.task,
            model=self.model_id_watercolor,
            model_revision='v1.0.0')
        output = pipe(
            {'text': 'sks style, a portrait painting of Johnny Depp'})
        cv2.imwrite('result_watercolor.png', output['output_imgs'][0])
        print('Image saved to result_watercolor.png')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_flat(self):
        pipe = pipeline(
            task=self.task, model=self.model_id_flat, model_revision='v1.0.0')
        output = pipe(
            {'text': 'sks style, a portrait painting of Johnny Depp'})
        cv2.imwrite('result_flat.png', output['output_imgs'][0])
        print('Image saved to result_flat.png')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_clipart(self):
        pipe = pipeline(
            task=self.task,
            model=self.model_id_clipart,
            model_revision='v1.0.0')
        output = pipe(
            {'text': 'archer style, a portrait painting of Johnny Depp'})
        cv2.imwrite('result_clipart.png', output['output_imgs'][0])
        print('Image saved to result_clipart.png')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_eulerasolver(self):
        from diffusers.schedulers import EulerAncestralDiscreteScheduler
        pipe = pipeline(
            task=self.task, model=self.model_id, model_revision='v1.0.0')
        pipe.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.pipeline.scheduler.config)
        output = pipe(
            {'text': 'sks style, a portrait painting of Johnny Depp'})
        cv2.imwrite('result_design2.png', output['output_imgs'][0])
        print('Image saved to result_design2.png')


if __name__ == '__main__':
    unittest.main()
