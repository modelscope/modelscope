# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import SpaceForDST
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import DialogStateTrackingPipeline
from modelscope.preprocessors import DialogStateTrackingPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.nlp.space.utils_dst import \
    tracking_and_print_dialog_states
from modelscope.utils.test_utils import test_level


class DialogStateTrackingTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.task_oriented_conversation
        self.model_id = 'damo/nlp_space_dialog-state-tracking'

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

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)

        model = SpaceForDST.from_pretrained(cache_path)
        preprocessor = DialogStateTrackingPreprocessor(model_dir=cache_path)
        pipelines = [
            DialogStateTrackingPipeline(
                model=model, preprocessor=preprocessor),
            pipeline(
                task=Tasks.task_oriented_conversation,
                model=model,
                preprocessor=preprocessor)
        ]
        tracking_and_print_dialog_states(self.test_case, pipelines)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)

        preprocessor = DialogStateTrackingPreprocessor(
            model_dir=model.model_dir)
        pipelines = [
            DialogStateTrackingPipeline(
                model=model, preprocessor=preprocessor),
            pipeline(task=self.task, model=model, preprocessor=preprocessor)
        ]

        tracking_and_print_dialog_states(self.test_case, pipelines)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipelines = [pipeline(task=self.task, model=self.model_id)]
        tracking_and_print_dialog_states(self.test_case, pipelines)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
