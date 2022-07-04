# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest
from typing import List

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model, SpaceForDialogStateTracking
from modelscope.pipelines import DialogStateTrackingPipeline, pipeline
from modelscope.preprocessors import DialogStateTrackingPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class DialogStateTrackingTest(unittest.TestCase):
    model_id = 'damo/nlp_space_dialog-state-tracking'
    test_case = [{
        'User-1':
        'Hi, I\'m looking for a train that is going to cambridge and arriving there by 20:45, '
        'is there anything like that?'
    }, {
        'System-1':
        'There are over 1,000 trains like that.  Where will you be departing from?',
        'Dialog_Act-1': {
            'Train-Inform': [['Choice', 'over 1'], ['Choice', '000']],
            'Train-Request': [['Depart', '?']]
        },
        'User-2': 'I am departing from birmingham new street.'
    }, {
        'System-2': 'Can you confirm your desired travel day?',
        'Dialog_Act-2': {
            'Train-Request': [['Day', '?']]
        },
        'User-3': 'I would like to leave on wednesday'
    }, {
        'System-3':
        'I show a train leaving birmingham new street at 17:40 and arriving at 20:23 on Wednesday.  '
        'Will this work for you?',
        'Dialog_Act-3': {
            'Train-Inform': [['Arrive', '20:23'], ['Leave', '17:40'],
                             ['Day', 'Wednesday'],
                             ['Depart', 'birmingham new street']]
        },
        'User-4':
        'That will, yes. Please make a booking for 5 people please.',
    }, {
        'System-4':
        'I\'ve booked your train tickets, and your reference number is A9NHSO9Y.',
        'Dialog_Act-4': {
            'Train-OfferBooked': [['Ref', 'A9NHSO9Y']]
        },
        'User-5':
        'Thanks so much. I would also need a place to say. '
        'I am looking for something with 4 stars and has free wifi.'
    }, {
        'System-5':
        'How about the cambridge belfry?  '
        'It has all the attributes you requested and a great name!  '
        'Maybe even a real belfry?',
        'Dialog_Act-5': {
            'Hotel-Recommend': [['Name', 'the cambridge belfry']]
        },
        'User-6':
        'That sounds great, could you make a booking for me please?',
    }, {
        'System-6':
        'What day would you like your booking for?',
        'Dialog_Act-6': {
            'Booking-Request': [['Day', '?']]
        },
        'User-7':
        'Please book it for Wednesday for 5 people and 5 nights, please.',
    }, {
        'System-7': 'Booking was successful. Reference number is : 5NAWGJDC.',
        'Dialog_Act-7': {
            'Booking-Book': [['Ref', '5NAWGJDC']]
        },
        'User-8': 'Thank you, goodbye',
    }]

    def tracking_and_print_dialog_states(
            self, pipelines: List[DialogStateTrackingPipeline]):
        import json
        pipelines_len = len(pipelines)
        history_states = [{}]
        utter = {}
        for step, item in enumerate(self.test_case):
            utter.update(item)
            result = pipelines[step % pipelines_len]({
                'utter':
                utter,
                'history_states':
                history_states
            })
            print(json.dumps(result))

            history_states.extend([result['dialog_states'], {}])

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)

        model = SpaceForDialogStateTracking(cache_path)
        preprocessor = DialogStateTrackingPreprocessor(model_dir=cache_path)
        pipelines = [
            DialogStateTrackingPipeline(
                model=model, preprocessor=preprocessor),
            pipeline(
                task=Tasks.dialog_state_tracking,
                model=model,
                preprocessor=preprocessor)
        ]
        self.tracking_and_print_dialog_states(pipelines)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        preprocessor = DialogStateTrackingPreprocessor(
            model_dir=model.model_dir)
        pipelines = [
            DialogStateTrackingPipeline(
                model=model, preprocessor=preprocessor),
            pipeline(
                task=Tasks.dialog_state_tracking,
                model=model,
                preprocessor=preprocessor)
        ]

        self.tracking_and_print_dialog_states(pipelines)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipelines = [
            pipeline(task=Tasks.dialog_state_tracking, model=self.model_id)
        ]
        self.tracking_and_print_dialog_states(pipelines)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipelines = [pipeline(task=Tasks.dialog_state_tracking)]
        self.tracking_and_print_dialog_states(pipelines)


if __name__ == '__main__':
    unittest.main()
