# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.pipelines.cv.anydoor_pipeline import AnydoorPipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class AnydoorTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.image_to_image_generation
        self.model_id = 'damo/AnyDoor_models'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run(self):
        ref_image = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_anydoor_fg.png'
        ref_mask = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_anydoor_fg_mask.png'
        bg_image = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_anydoor_bg.jpg'
        bg_mask = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_anydoor_bg_mask.png'
        save_path = 'data/test/images/image_anydoor_gen.png'

        anydoor_pipline: AnydoorPipeline = pipeline(
            self.task, model=self.model_id)
        out = anydoor_pipline((ref_image, ref_mask, bg_image, bg_mask))
        image = out['output_img']
        image.save(save_path)


if __name__ == '__main__':
    unittest.main()
