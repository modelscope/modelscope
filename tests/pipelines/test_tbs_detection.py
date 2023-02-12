import unittest
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class ObjectDetectionTest(unittest.TestCase, DemoCompatibilityCheck):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        tbs_detect = pipeline(Tasks.image_object_detection, model='landingAI/LD_CytoBrainCerv')
        outputs = tbs_detect(input='data/test/images/tbs_detection.jpg')
        
        
if __name__ == '__main__':
    unittest.main()
    