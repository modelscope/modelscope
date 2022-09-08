# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from PIL import Image

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class ImageReidPersonTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.input_location = 'data/test/images/image_reid_person.jpg'
        self.model_id = 'damo/cv_passvitb_image-reid-person_market'
        self.task = Tasks.image_reid_person

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_image_reid_person(self):
        image_reid_person = pipeline(
            Tasks.image_reid_person, model=self.model_id)
        result = image_reid_person(self.input_location)
        assert result and OutputKeys.IMG_EMBEDDING in result
        print(
            f'The shape of img embedding is: {result[OutputKeys.IMG_EMBEDDING].shape}'
        )
        print(f'The img embedding is: {result[OutputKeys.IMG_EMBEDDING]}')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_image_reid_person_with_image(self):
        image_reid_person = pipeline(
            Tasks.image_reid_person, model=self.model_id)
        img = Image.open(self.input_location)
        result = image_reid_person(img)
        assert result and OutputKeys.IMG_EMBEDDING in result
        print(
            f'The shape of img embedding is: {result[OutputKeys.IMG_EMBEDDING].shape}'
        )
        print(f'The img embedding is: {result[OutputKeys.IMG_EMBEDDING]}')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_image_reid_person_with_default_model(self):
        image_reid_person = pipeline(Tasks.image_reid_person)
        result = image_reid_person(self.input_location)
        assert result and OutputKeys.IMG_EMBEDDING in result
        print(
            f'The shape of img embedding is: {result[OutputKeys.IMG_EMBEDDING].shape}'
        )
        print(f'The img embedding is: {result[OutputKeys.IMG_EMBEDDING]}')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
