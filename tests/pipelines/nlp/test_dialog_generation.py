# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
import tempfile
import unittest

from tests.case.nlp.dialog_generation_case import test_case

from maas_lib.models.nlp import DialogGenerationModel
from maas_lib.pipelines import DialogGenerationPipeline, pipeline
from maas_lib.preprocessors import DialogGenerationPreprocessor


def merge(info, result):
    return info


class DialogGenerationTest(unittest.TestCase):

    def test_run(self):

        modeldir = '/Users/yangliu/Desktop/space-dialog-generation'

        preprocessor = DialogGenerationPreprocessor(model_dir=modeldir)
        model = DialogGenerationModel(
            model_dir=modeldir,
            text_field=preprocessor.text_field,
            config=preprocessor.config)
        print(model.forward(None))
        # pipeline = DialogGenerationPipeline(model=model, preprocessor=preprocessor)
        #
        # history_dialog_info = {}
        # for step, item in enumerate(test_case['sng0073']['log']):
        #     user_question = item['user']
        #     print('user: {}'.format(user_question))
        #
        #     # history_dialog_info = merge(history_dialog_info,
        #     #                             result) if step > 0 else {}
        #     result = pipeline(user_question, history=history_dialog_info)
        #     #
        #     # print('sys : {}'.format(result['pred_answer']))


if __name__ == '__main__':
    unittest.main()
