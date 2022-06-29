# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
import tempfile
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import DialogStateTrackingModel
from modelscope.pipelines import DialogStateTrackingPipeline, pipeline
from modelscope.preprocessors import DialogStateTrackingPreprocessor
from modelscope.utils.constant import Tasks


class DialogStateTrackingTest(unittest.TestCase):
    model_id = 'damo/nlp_space_dialog-state-tracking'

    test_case = [{
        'utter': {
            'User-1':
            'am looking for a place to to stay that has cheap price range it should be in a type of hotel'
        },
        'history_states': [{}]
    }, {
        'utter': {
            'User-1':
            'am looking for a place to to stay that has cheap price range it should be in a type of hotel',
            'System-1':
            'Okay, do you have a specific area you want to stay in?',
            'Dialog_Act-1': {
                'Hotel-Request': [['Area', '?']]
            },
            'User-2':
            "no, i just need to make sure it's cheap. oh, and i need parking"
        },
        'history_states': [{}, {
            'taxi': {
                'book': {
                    'booked': []
                },
                'semi': {
                    'leaveAt': '',
                    'destination': '',
                    'departure': '',
                    'arriveBy': ''
                }
            },
            'police': {
                'book': {
                    'booked': []
                },
                'semi': {}
            },
            'restaurant': {
                'book': {
                    'booked': [],
                    'people': '',
                    'day': '',
                    'time': ''
                },
                'semi': {
                    'food': '',
                    'pricerange': '',
                    'name': '',
                    'area': ''
                }
            },
            'hospital': {
                'book': {
                    'booked': []
                },
                'semi': {
                    'department': ''
                }
            },
            'hotel': {
                'book': {
                    'booked': [],
                    'people': '',
                    'day': '',
                    'stay': ''
                },
                'semi': {
                    'name': 'not mentioned',
                    'area': 'not mentioned',
                    'parking': 'not mentioned',
                    'pricerange': 'cheap',
                    'stars': 'not mentioned',
                    'internet': 'not mentioned',
                    'type': 'hotel'
                }
            },
            'attraction': {
                'book': {
                    'booked': []
                },
                'semi': {
                    'type': '',
                    'name': '',
                    'area': ''
                }
            },
            'train': {
                'book': {
                    'booked': [],
                    'people': ''
                },
                'semi': {
                    'leaveAt': '',
                    'destination': '',
                    'day': '',
                    'arriveBy': '',
                    'departure': ''
                }
            }
        }, {}]
    }]

    def test_run(self):
        cache_path = '/Users/yangliu/Space/maas_model/nlp_space_dialog-state-tracking'
        # cache_path = snapshot_download(self.model_id)

        model = DialogStateTrackingModel(cache_path)
        preprocessor = DialogStateTrackingPreprocessor(model_dir=cache_path)
        pipelines = [
            DialogStateTrackingPipeline(
                model=model, preprocessor=preprocessor),
            # pipeline(
            #     task=Tasks.dialog_state_tracking,
            #     model=model,
            #     preprocessor=preprocessor)
        ]

        history_states = {}
        pipelines_len = len(pipelines)
        for step, item in enumerate(self.test_case):
            history_states = pipelines[step % pipelines_len](item)
            print(history_states)

    @unittest.skip('test with snapshot_download')
    def test_run_with_model_from_modelhub(self):
        pass


if __name__ == '__main__':
    unittest.main()
