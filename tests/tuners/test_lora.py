# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

import numpy as np
import torch

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models.base import Model
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.trainers import build_trainer
from modelscope.tuners.lora import (Linear, LoRATuner,
                                    mark_only_lora_as_trainable)
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.test_utils import test_level


class TestLora(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip in this level')
    def test_lora_base(self):

        class TestModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.lora = Linear(16, 16, r=4)

        model = TestModel()
        mark_only_lora_as_trainable(model)
        model.train()
        loss = model.lora(torch.ones(16, 16))
        loss = loss.sum()
        loss.backward()

        model = TestModel()
        mark_only_lora_as_trainable(model)
        model.eval()
        loss = model.lora(torch.ones(16, 16))
        loss = loss.sum()
        try:
            loss.backward()
        except Exception:
            pass
        else:
            raise Exception('No tensor needs grad, should throw en error here')

    @unittest.skipUnless(test_level() >= 0, 'skip in this level')
    def test_lora_smoke_test(self):
        dataset = MsDataset.load(
            'clue', subset_name='afqmc',
            split='train').to_hf_dataset().select(range(2))

        model_dir = snapshot_download(
            'damo/nlp_structbert_sentence-similarity_chinese-tiny')
        model = Model.from_pretrained(
            'damo/nlp_structbert_sentence-similarity_chinese-tiny',
            adv_grad_factor=None)

        cfg_file = os.path.join(model_dir, 'configuration.json')

        kwargs = dict(
            model=model,
            cfg_file=cfg_file,
            train_dataset=dataset,
            eval_dataset=dataset,
            work_dir=self.tmp_dir,
            efficient_tuners=[{
                'type': 'lora',
                'replace_modules': ['query', 'key', 'value']
            }])

        trainer = build_trainer(default_args=kwargs)
        trainer.train()
        output_dir = os.path.join(self.tmp_dir, ModelFile.TRAIN_OUTPUT_DIR)

        def pipeline_sentence_similarity(model_dir):
            model = Model.from_pretrained(model_dir)
            LoRATuner.tune(model, replace_modules=['query', 'key', 'value'])
            model.load_state_dict(
                torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
            model.eval()
            pipeline_ins = pipeline(
                task=Tasks.sentence_similarity, model=model)
            return pipeline_ins(input=('test', 'this is a test'))

        output1 = pipeline_sentence_similarity(
            'damo/nlp_structbert_sentence-similarity_chinese-tiny')

        LoRATuner.unpatch_lora(model, ['query', 'key', 'value'])
        model.save_pretrained(
            output_dir, save_checkpoint_names='pytorch_model.bin')

        def pipeline_sentence_similarity_origin():
            model = Model.from_pretrained(output_dir)
            model.eval()
            pipeline_ins = pipeline(
                task=Tasks.sentence_similarity, model=model)
            return pipeline_ins(input=('test', 'this is a test'))

        output2 = pipeline_sentence_similarity_origin()
        print(output1, output2)
        self.assertTrue(all(np.isclose(output1['scores'], output2['scores'])))


if __name__ == '__main__':
    unittest.main()
