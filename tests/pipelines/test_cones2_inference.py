# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import cv2

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ConesStableDiffusionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.text_to_image_synthesis
        self.model_id = 'damo/Cones2'

    @unittest.skipUnless(test_level() >= 2,
                         'skip test for diffusers compatible')
    def test_run(self):

        pipe = pipeline(
            task=self.task, model=self.model_id, model_revision='v1.0.1')
        output = pipe({
            'text': 'a mug and a dog on the beach',
            'subject_list': [['mug', 2], ['dog', 5]],
            'color_context': {
                '255,192,0': ['mug', 2.5],
                '255,0,0': ['dog', 2.5]
            },
            'layout': 'data/test/images/mask_example.png'
        })
        cv2.imwrite('result.png', output['output_imgs'][0])
        print('Image saved to result.png')


if __name__ == '__main__':
    unittest.main()
