import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class MultiModalFeatureTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run(self):
        animal_recog = pipeline(
            Tasks.image_classification,
            model='damo/cv_resnest101_animal_recognition')
        result = animal_recog('data/test/images/image1.jpg')
        print(result)


if __name__ == '__main__':
    unittest.main()
