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
            "I'm looking for a place to stay. It needs to be a guesthouse and include free wifi."
        },
        'history_states': [{}]
    }]

    def test_run(self):
        cache_path = '/Users/yangliu/Space/maas_model/nlp_space_dialog-state-tracking'
        # cache_path = snapshot_download(self.model_id)

        model = DialogStateTrackingModel(cache_path)
        preprocessor = DialogStateTrackingPreprocessor(model_dir=cache_path)
        pipeline1 = DialogStateTrackingPipeline(
            model=model, preprocessor=preprocessor)

        history_states = {}
        for step, item in enumerate(self.test_case):
            history_states = pipeline1(item)
            print(history_states)

    @unittest.skip('test with snapshot_download')
    def test_run_with_model_from_modelhub(self):
        pass


if __name__ == '__main__':
    unittest.main()
