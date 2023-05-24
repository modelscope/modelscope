# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import unittest

from PIL import Image

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import masks_visualization
from modelscope.utils.test_utils import test_level


class VideoObjectSegmentationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = 'video-object-segmentation'
        self.model_id = 'damo/cv_rdevos_video-object-segmentation'
        self.input_location = 'data/test/videos/video_object_segmentation_test'
        self.images_dir = os.path.join(self.input_location, 'JPEGImages')
        self.mask_file = os.path.join(self.input_location, 'Annotations',
                                      '00000.png')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_video_object_segmentation(self):
        input_images = []
        for image_file in sorted(os.listdir(self.images_dir)):
            img = Image.open(os.path.join(self.images_dir, image_file))\
                .convert('RGB')
            input_images.append(img)
        mask = Image.open(self.mask_file).convert('P')
        input = {'images': input_images, 'mask': mask}

        segmentor = pipeline(
            Tasks.video_object_segmentation, model=self.model_id)
        result = segmentor(input)
        out_masks = result[OutputKeys.MASKS]

        vis_masks = masks_visualization(out_masks, mask.getpalette())

        os.makedirs('test_result', exist_ok=True)
        for f, vis_mask in enumerate(vis_masks):
            vis_mask.save(os.path.join('test_result', '{:05d}.png'.format(f)))

        print('test_video_object_segmentation DONE')


if __name__ == '__main__':
    unittest.main()
