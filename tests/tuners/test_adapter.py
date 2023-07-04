# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

import numpy as np
import torch

from modelscope import read_config
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models.base import Model
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.swift import Swift
from modelscope.swift.adapter import AdapterConfig
from modelscope.trainers import build_trainer
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.test_utils import test_level


class TestAdapter(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip in this level')
    def test_adapter_smoke_test(self):
        dataset = MsDataset.load(
            'clue', subset_name='afqmc',
            split='train').to_hf_dataset().select(range(2))

        model_dir = snapshot_download(
            'damo/nlp_structbert_sentence-similarity_chinese-tiny')
        model = Model.from_pretrained(model_dir, adv_grad_factor=None)

        cfg_file = os.path.join(model_dir, 'configuration.json')

        model_cfg = os.path.join(model_dir, 'config.json')
        model_cfg = read_config(model_cfg)

        adapter_config = AdapterConfig(
            dim=model_cfg.hidden_size,
            module_name=r'.*layer\.\d+$',
            method_name='feed_forward_chunk',
            hidden_pos=0)
        model = Swift.prepare_model(model, adapter_config)
        kwargs = dict(
            model=model,
            cfg_file=cfg_file,
            train_dataset=dataset,
            eval_dataset=dataset,
            work_dir=self.tmp_dir)

        trainer = build_trainer(default_args=kwargs)
        trainer.train()
        output_dir = os.path.join(self.tmp_dir, ModelFile.TRAIN_OUTPUT_DIR)

        def pipeline_sentence_similarity(model_dir):
            model = Model.from_pretrained(model_dir)
            adapter_config.pretrained_weights = output_dir
            Swift.prepare_model(model, adapter_config)
            model.eval()
            pipeline_ins = pipeline(
                task=Tasks.sentence_similarity, model=model)
            return pipeline_ins(input=('test', 'this is a test'))

        output1 = pipeline_sentence_similarity(
            'damo/nlp_structbert_sentence-similarity_chinese-tiny')
        print(output1)


if __name__ == '__main__':
    unittest.main()
