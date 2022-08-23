# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from torchvision.utils import save_image

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class Image2ImageGenerationTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        r"""We provide two generation modes, i.e., Similar Image Generation and Interpolation.
            You can pass the following parameters for different mode.
            1. Similar Image Generation Mode:
            2. Interpolation Mode:
        """
        img2img_gen_pipeline = pipeline(
            Tasks.image_to_image_generation,
            model='damo/cv_latent_diffusion_image2image_generate')

        # Similar Image Generation mode
        result1 = img2img_gen_pipeline('data/test/images/img2img_input.jpg')
        # Interpolation Mode
        result2 = img2img_gen_pipeline(('data/test/images/img2img_input.jpg',
                                        'data/test/images/img2img_style.jpg'))
        save_image(
            result1[OutputKeys.OUTPUT_IMG].clamp(-1, 1),
            'result1.jpg',
            range=(-1, 1),
            normalize=True,
            nrow=4)
        save_image(
            result2[OutputKeys.OUTPUT_IMG].clamp(-1, 1),
            'result2.jpg',
            range=(-1, 1),
            normalize=True,
            nrow=4)


if __name__ == '__main__':
    unittest.main()
