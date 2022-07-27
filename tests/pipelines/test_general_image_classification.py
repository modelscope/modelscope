import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class GeneralImageClassificationTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_ImageNet(self):
        general_image_classification = pipeline(
            Tasks.image_classification_imagenet,
            model='damo/cv_vit-base_image-classification_ImageNet-labels')
        result = general_image_classification('data/test/images/bird.JPEG')
        print(result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_Dailylife(self):
        general_image_classification = pipeline(
            Tasks.image_classification_dailylife,
            model='damo/cv_vit-base_image-classification_Dailylife-labels')
        result = general_image_classification('data/test/images/bird.JPEG')
        print(result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_ImageNet_default_task(self):
        general_image_classification = pipeline(
            Tasks.image_classification_imagenet)
        result = general_image_classification('data/test/images/bird.JPEG')
        print(result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_Dailylife_default_task(self):
        general_image_classification = pipeline(
            Tasks.image_classification_dailylife)
        result = general_image_classification('data/test/images/bird.JPEG')
        print(result)


if __name__ == '__main__':
    unittest.main()
