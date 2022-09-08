import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class AnimalRecognitionTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.animal_recognition
        self.model_id = 'damo/cv_resnest101_animal_recognition'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run(self):
        animal_recognition = pipeline(
            Tasks.animal_recognition, model=self.model_id)
        result = animal_recognition('data/test/images/dogs.jpg')
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
