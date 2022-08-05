# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest
from typing import List

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import SpaceForDialogModeling
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import TaskOrientedConversationPipeline
from modelscope.preprocessors import DialogModelingPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class TaskOrientedConversationTest(unittest.TestCase):
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

    def generate_and_print_dialog_response(
            self, pipelines: List[TaskOrientedConversationPipeline]):

        result = {}
        for step, item in enumerate(self.test_case['sng0073']['log']):
            user = item['user']
            print('user: {}'.format(user))

            result = pipelines[step % 2]({
                'user_input': user,
                'history': result
            })
            print('response : {}'.format(result['response']))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):

        cache_path = snapshot_download(self.model_id)

        preprocessor = DialogModelingPreprocessor(model_dir=cache_path)
        model = SpaceForDialogModeling(
            model_dir=cache_path,
            text_field=preprocessor.text_field,
            config=preprocessor.config)
        pipelines = [
            TaskOrientedConversationPipeline(
                model=model, preprocessor=preprocessor),
            pipeline(
                task=Tasks.task_oriented_conversation,
                model=model,
                preprocessor=preprocessor)
        ]
        self.generate_and_print_dialog_response(pipelines)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        preprocessor = DialogModelingPreprocessor(model_dir=model.model_dir)

        pipelines = [
            TaskOrientedConversationPipeline(
                model=model, preprocessor=preprocessor),
            pipeline(
                task=Tasks.task_oriented_conversation,
                model=model,
                preprocessor=preprocessor)
        ]

        self.generate_and_print_dialog_response(pipelines)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipelines = [
            pipeline(
                task=Tasks.task_oriented_conversation, model=self.model_id),
            pipeline(
                task=Tasks.task_oriented_conversation, model=self.model_id)
        ]
        self.generate_and_print_dialog_response(pipelines)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipelines = [
            pipeline(task=Tasks.task_oriented_conversation),
            pipeline(task=Tasks.task_oriented_conversation)
        ]
        self.generate_and_print_dialog_response(pipelines)


if __name__ == '__main__':
    unittest.main()
