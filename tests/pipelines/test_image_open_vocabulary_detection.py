# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import cv2

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import draw_box
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


class ImageOpenVocabularyDetectionTest(unittest.TestCase):

    def setUp(self) -> None:
        os.system(
            'pip install tensorflow==2.9.2 -i https://pypi.tuna.tsinghua.edu.cn/simple'
        )
        logger.info('upgrade tensorflow finished')

        self.task = Tasks.open_vocabulary_detection
        self.model_id = 'damo/cv_resnet152_open-vocabulary-detection_vild'
        self.image = 'data/test/images/image_open_vocabulary_detection.jpg'
        self.category_names = ';'.join([
            'flipflop', 'street sign', 'bracelet', 'necklace', 'shorts',
            'floral camisole', 'orange shirt', 'purple dress', 'yellow tee',
            'green umbrella', 'pink striped umbrella', 'transparent umbrella',
            'plain pink umbrella', 'blue patterned umbrella', 'koala',
            'electric box', 'car', 'pole'
        ])
        self.input = {'img': self.image, 'category_names': self.category_names}

    def tearDown(self) -> None:
        os.system(
            'pip install tensorflow-gpu==1.15 -i https://pypi.tuna.tsinghua.edu.cn/simple'
        )
        logger.info('degrade tensorflow finished')
        return super().tearDown()

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        vild_pipeline = pipeline(task=self.task, model=model)
        result = vild_pipeline(input=self.input)
        image = cv2.imread(self.image)
        draw_box(image, result[OutputKeys.BOXES][0, :])
        cv2.imwrite('result_modelhub.jpg', image)
        print('Test run with model from modelhub ok.')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_name(self):
        vild_pipeline = pipeline(task=self.task, model=self.model_id)
        result = vild_pipeline(self.input)
        image = cv2.imread(self.image)
        draw_box(image, result[OutputKeys.BOXES][0, :])
        cv2.imwrite('result_modelname.jpg', image)
        print('Test run with model name ok.')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        vild_pipeline = pipeline(self.task, model=cache_path)
        result = vild_pipeline(input=self.input)
        image = cv2.imread(self.image)
        draw_box(image, result[OutputKeys.BOXES][0, :])
        cv2.imwrite('result_snapshot.jpg', image)
        print('Test run with snapshot ok.')


if __name__ == '__main__':
    unittest.main()
