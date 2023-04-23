# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.preprocessors import DialogueClassificationUsePreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class UserSatisfactionEstimationTest(unittest.TestCase):

    model_id = 'damo/nlp_user-satisfaction-estimation_chinese'
    input_dialogue = [('返修退换货咨询|||', '手机有质量问题怎么办|||稍等，我看下', '开不开机了|||',
                       '说话|||很好')]

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        preprocessor = DialogueClassificationUsePreprocessor(model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.text_classification,
            model=model,
            preprocessor=preprocessor)
        print(pipeline_ins(input=self.input_dialogue))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.text_classification, model=self.model_id)
        print(pipeline_ins(input=self.input_dialogue))


if __name__ == '__main__':
    unittest.main()
