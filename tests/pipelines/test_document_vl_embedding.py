# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
import unittest

import json

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class DocumentVLEmbeddingTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/multi-modal_convnext-roberta-base_vldoc-embedding'
        cache_path = snapshot_download(self.model_id)
        self.test_image = osp.join(cache_path, 'data/demo.png')
        self.test_json = osp.join(cache_path, 'data/demo.json')
        self.task = Tasks.document_vl_embedding

    def pipeline_inference(self, pipe: Pipeline):
        inp = {'images': [self.test_image], 'ocr_info_paths': [self.test_json]}
        result = pipe(inp)

        print('Results of VLDoc: ')
        for k, v in result.items():
            print(f'{k}: {v.size()}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        doc_VL_emb_pipeline = pipeline(task=self.task, model=self.model_id)
        self.pipeline_inference(doc_VL_emb_pipeline)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        print('test_run_with_model_from_modelhub')
        model = Model.from_pretrained(self.model_id)

        doc_VL_emb_pipeline = pipeline(task=self.task, model=model)
        self.pipeline_inference(doc_VL_emb_pipeline)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        print('test_run_modelhub_default_model')
        # default model: VLDoc
        vldoc_doc_VL_emb_pipeline = pipeline(self.task)
        self.pipeline_inference(vldoc_doc_VL_emb_pipeline)


if __name__ == '__main__':
    unittest.main()
