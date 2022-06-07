# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
import tempfile
import unittest

from maas_lib.fileio import File
from maas_lib.models.nlp import DialogGenerationModel
from maas_lib.pipelines import DialogGenerationPipeline, pipeline
from maas_lib.preprocessors import DialogGenerationPreprocessor
from maas_lib.utils.constant import Tasks

dialog_case = [{
    'user':
    'am looking for a place to to stay that has cheap price range it should be in a type of hotel',
    'sys':
    'okay , do you have a specific area you want to stay in ?'
}, {
    'user':
    'no , i just need to make sure it is cheap . oh , and i need parking',
    'sys':
    'i found 1 cheap hotel for you that include -s parking . do you like me to book it ?'
}, {
    'user':
    'yes , please . 6 people 3 nights starting on tuesday .',
    'sys':
    "i am sorry but i was n't able to book that for you for tuesday . is there another day you would like "
    'to stay or perhaps a shorter stay ? '
}, {
    'user':
    'how about only 2 nights .',
    'sys':
    'booking was successful . reference number is : 7gawk763 . anything else i can do for you ?',
}, {
    'user': 'no , that will be all . goodbye .',
    'sys': 'thank you for using our services .'
}]


class DialogGenerationTest(unittest.TestCase):

    def test_run(self):
        for item in dialog_case:
            q = item['user']
            a = item['sys']
            print('user:{}'.format(q))
            print('sys:{}'.format(a))

        # preprocessor = DialogGenerationPreprocessor()
        # # data = DialogGenerationData()
        # model = DialogGenerationModel(path, preprocessor.tokenizer)
        # pipeline = DialogGenerationPipeline(model, preprocessor)
        #
        # history_dialog = []
        # for item in dialog_case:
        #     user_question = item['user']
        #     print('user: {}'.format(user_question))
        #
        # pipeline(user_question)
        #
        # sys_answer, history_dialog = pipeline()
        #
        # print('sys : {}'.format(sys_answer))


if __name__ == '__main__':
    unittest.main()
