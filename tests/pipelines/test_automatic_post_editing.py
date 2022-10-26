# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class AutomaticPostEditingTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.translation
        self.model_id = 'damo/nlp_automatic_post_editing_for_translation_en2de'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_for_en2de(self):
        inputs = 'Simultaneously, the Legion took part to the pacification of Algeria, plagued by various tribal ' \
                 'rebellions and razzias.\005Gleichzeitig nahm die Legion an der Befriedung Algeriens teil, die von ' \
                 'verschiedenen Stammesaufständen und Rasias heimgesucht wurde.'
        pipeline_ins = pipeline(self.task, model=self.model_id)
        print(pipeline_ins(input=inputs))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
