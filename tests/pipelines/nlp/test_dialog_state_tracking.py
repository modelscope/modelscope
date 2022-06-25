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
    test_case = {}

    def test_run(self):
        # cache_path = ''
        # cache_path = snapshot_download(self.model_id)

        # preprocessor = DialogStateTrackingPreprocessor(model_dir=cache_path)
        # model = DialogStateTrackingModel(
        #     model_dir=cache_path,
        #     text_field=preprocessor.text_field,
        #     config=preprocessor.config)
        # pipelines = [
        #     DialogStateTrackingPipeline(model=model, preprocessor=preprocessor),
        #     pipeline(
        #         task=Tasks.dialog_modeling,
        #         model=model,
        #         preprocessor=preprocessor)
        # ]

        print('jizhu test')

    @unittest.skip('test with snapshot_download')
    def test_run_with_model_from_modelhub(self):
        pass


if __name__ == '__main__':
    unittest.main()
