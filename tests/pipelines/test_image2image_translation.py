# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class Image2ImageTranslationTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        r"""We provide three translation modes, i.e., uncropping, colorization and combination.
            You can pass the following parameters for different mode.
            1. Uncropping Mode:
            result = img2img_gen_pipeline(('data/test/images/img2img_input.jpg', 'left', 0, 'result.jpg'))
            2. Colorization Mode:
            result = img2img_gen_pipeline(('data/test/images/img2img_input.jpg', 1, 'result.jpg'))
            3. Combination Mode:
            just like the following code.
        """
        img2img_gen_pipeline = pipeline(
            Tasks.image_to_image_translation,
            model='damo/cv_latent_diffusion_image2image_translation')
        result = img2img_gen_pipeline(
            ('data/test/images/img2img_input_mask.png',
             'data/test/images/img2img_input_masked_img.png', 2,
             'result.jpg'))  # combination mode

        print(f'output: {result}.')


if __name__ == '__main__':
    unittest.main()
