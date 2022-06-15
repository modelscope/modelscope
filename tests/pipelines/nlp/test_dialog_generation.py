# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
import tempfile
import unittest

from modelscope.models.nlp import DialogGenerationModel
from modelscope.pipelines import DialogGenerationPipeline, pipeline
from modelscope.preprocessors import DialogGenerationPreprocessor


def merge(info, result):
    return info


class DialogGenerationTest(unittest.TestCase):
    test_case = {
        'sng0073': {
            'goal': {
                'taxi': {
                    'info': {
                        'leaveat': '17:15',
                        'destination': 'pizza hut fen ditton',
                        'departure': "saint john's college"
                    },
                    'reqt': ['car', 'phone'],
                    'fail_info': {}
                }
            },
            'log': [{
                'user':
                "i would like a taxi from saint john 's college to pizza hut fen ditton .",
                'user_delex':
                'i would like a taxi from [value_departure] to [value_destination] .',
                'resp':
                'what time do you want to leave and what time do you want to arrive by ?',
                'sys':
                'what time do you want to leave and what time do you want to arrive by ?',
                'pointer': '0,0,0,0,0,0',
                'match': '',
                'constraint':
                "[taxi] destination pizza hut fen ditton departure saint john 's college",
                'cons_delex': '[taxi] destination departure',
                'sys_act': '[taxi] [request] leave arrive',
                'turn_num': 0,
                'turn_domain': '[taxi]'
            }, {
                'user': 'i want to leave after 17:15 .',
                'user_delex': 'i want to leave after [value_leave] .',
                'resp':
                'booking completed ! your taxi will be [value_car] contact number is [value_phone]',
                'sys':
                'booking completed ! your taxi will be blue honda contact number is 07218068540',
                'pointer': '0,0,0,0,0,0',
                'match': '',
                'constraint':
                "[taxi] destination pizza hut fen ditton departure saint john 's college leave 17:15",
                'cons_delex': '[taxi] destination departure leave',
                'sys_act': '[taxi] [inform] car phone',
                'turn_num': 1,
                'turn_domain': '[taxi]'
            }, {
                'user': 'thank you for all the help ! i appreciate it .',
                'user_delex': 'thank you for all the help ! i appreciate it .',
                'resp':
                'you are welcome . is there anything else i can help you with today ?',
                'sys':
                'you are welcome . is there anything else i can help you with today ?',
                'pointer': '0,0,0,0,0,0',
                'match': '',
                'constraint':
                "[taxi] destination pizza hut fen ditton departure saint john 's college leave 17:15",
                'cons_delex': '[taxi] destination departure leave',
                'sys_act': '[general] [reqmore]',
                'turn_num': 2,
                'turn_domain': '[general]'
            }, {
                'user': 'no , i am all set . have a nice day . bye .',
                'user_delex': 'no , i am all set . have a nice day . bye .',
                'resp': 'you too ! thank you',
                'sys': 'you too ! thank you',
                'pointer': '0,0,0,0,0,0',
                'match': '',
                'constraint':
                "[taxi] destination pizza hut fen ditton departure saint john 's college leave 17:15",
                'cons_delex': '[taxi] destination departure leave',
                'sys_act': '[general] [bye]',
                'turn_num': 3,
                'turn_domain': '[general]'
            }]
        }
    }

    def test_run(self):

        modeldir = '/Users/yangliu/Desktop/space-dialog-generation'

        preprocessor = DialogGenerationPreprocessor(model_dir=modeldir)
        model = DialogGenerationModel(
            model_dir=modeldir,
            text_field=preprocessor.text_field,
            config=preprocessor.config)
        print(model.forward(None))
        # pipeline = DialogGenerationPipeline(
        #     model=model, preprocessor=preprocessor)

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
        print('test')


if __name__ == '__main__':
    unittest.main()
