# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class DDPMImageSemanticSegmentationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.image_segmentation
        self.model_id = 'damo/cv_diffusion_image-segmentation'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    @unittest.skipIf(not torch.cuda.is_available(), 'cuda unittest')
    def test_ddpm_image_semantic_segmentation(self):
        input_location = 'data/test/images/image_ffhq34_00041527.png'

        pp = pipeline(Tasks.semantic_segmentation, model=self.model_id)
        result = pp(input_location)
        if result:
            print(result)
        else:
            raise ValueError('process error')


if __name__ == '__main__':
    unittest.main()
