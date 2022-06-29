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
        'User-1':
        'am looking for a place to to stay that has cheap price range it should be in a type of hotel'
    }, {
        'System-1':
        'Okay, do you have a specific area you want to stay in?',
        'Dialog_Act-1': {
            'Hotel-Request': [['Area', '?']]
        },
        'User-2':
        "no, i just need to make sure it's cheap. oh, and i need parking"
    }, {
        'System-2':
        'I found 1 cheap hotel for you that includes parking. Do you like me to book it?',
        'Dialog_Act-2': {
            'Booking-Inform': [['none', 'none']],
            'Hotel-Inform': [['Price', 'cheap'], ['Choice', '1'],
                             ['Parking', 'none']]
        },
        'User-3':
        'Yes, please. 6 people 3 nights starting on tuesday.'
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

        history_states = [{}]
        utter = {}
        pipelines_len = len(pipelines)
        for step, item in enumerate(self.test_case):
            utter.update(item)
            ds = pipelines[step % pipelines_len]({
                'utter':
                utter,
                'history_states':
                history_states
            })
            print(ds)

            history_states.extend([ds, {}])

    @unittest.skip('test with snapshot_download')
    def test_run_with_model_from_modelhub(self):
        pass


if __name__ == '__main__':
    unittest.main()
