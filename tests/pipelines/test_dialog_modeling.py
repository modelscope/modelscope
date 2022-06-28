# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
import tempfile
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import SpaceForDialogModelingModel
from modelscope.pipelines import DialogModelingPipeline, pipeline
from modelscope.preprocessors import DialogModelingPreprocessor
from modelscope.utils.constant import Tasks


class DialogModelingTest(unittest.TestCase):
    model_id = 'damo/nlp_space_dialog-modeling'
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

    @unittest.skip('test with snapshot_download')
    def test_run(self):

        cache_path = snapshot_download(self.model_id)

        preprocessor = DialogModelingPreprocessor(model_dir=cache_path)
        model = SpaceForDialogModelingModel(
            model_dir=cache_path,
            text_field=preprocessor.text_field,
            config=preprocessor.config)
        pipelines = [
            DialogModelingPipeline(model=model, preprocessor=preprocessor),
            pipeline(
                task=Tasks.dialog_modeling,
                model=model,
                preprocessor=preprocessor)
        ]

        result = {}
        for step, item in enumerate(self.test_case['sng0073']['log']):
            user = item['user']
            print('user: {}'.format(user))

            result = pipelines[step % 2]({
                'user_input': user,
                'history': result
            })
            print('sys : {}'.format(result['sys']))

    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        preprocessor = DialogModelingPreprocessor(model_dir=model.model_dir)

        pipelines = [
            DialogModelingPipeline(model=model, preprocessor=preprocessor),
            pipeline(
                task=Tasks.dialog_modeling,
                model=model,
                preprocessor=preprocessor)
        ]

        result = {}
        for step, item in enumerate(self.test_case['sng0073']['log']):
            user = item['user']
            print('user: {}'.format(user))

            result = pipelines[step % 2]({
                'user_input': user,
                'history': result
            })
            print('sys : {}'.format(result['sys']))


if __name__ == '__main__':
    unittest.main()
