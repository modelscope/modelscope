# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import cv2
import numpy as np

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ActionDetectionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.image_to_image_generation
        self.model_id = 'damo/AnyDoor'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run(self):
        reference_image_path = 'data/test/images/image_anydoor_fg.png'
        bg_image_path = 'data/test/images/image_anydoor_bg.png'
        bg_mask_path = 'data/test/images/image_anydoor_bg_mask.png'
        save_path = 'data/test/images/image_anydoor_gen.png'

        image = cv2.imread(reference_image_path, cv2.IMREAD_UNCHANGED)
        mask = (image[:, :, -1] > 128).astype(np.uint8)
        image = image[:, :, :-1]
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        ref_image = image
        ref_mask = mask

        # background image
        back_image = cv2.imread(bg_image_path).astype(np.uint8)
        back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)

        # background mask
        tar_mask = cv2.imread(bg_mask_path)[:, :, 0] > 128
        tar_mask = tar_mask.astype(np.uint8)

        anydoor_pipline = pipeline(self.task, model=self.model_id)
        gen_image = anydoor_pipline(
            (ref_image, ref_mask, back_image.copy(), tar_mask))
        cv2.imwrite(save_path, gen_image)


if __name__ == '__main__':
    unittest.main()
