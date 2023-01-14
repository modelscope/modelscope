# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from collections.abc import Mapping
from typing import Any, Dict, List

import torch
from megatron_util import mpu

from modelscope.metainfo import Trainers
from modelscope.models import TorchModel
from modelscope.models.nlp import GPT3ForTextGeneration
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.nlp_trainer import NlpEpochBasedTrainer
from modelscope.utils.config import Config


@TRAINERS.register_module(module_name=Trainers.gpt3_trainer)
class GPT3Trainer(NlpEpochBasedTrainer):

    def rebuild_config(self, cfg: Config):
        cfg = super().rebuild_config(cfg)
        cfg.model.rank = int(os.environ.get('RANK', 0))
        return cfg

    def train_step(self, model: TorchModel, inputs: Mapping):
        keys = list(inputs.keys())
        datatype = torch.int64
        inputs = mpu.broadcast_data(keys, inputs, datatype)
        return super().train_step(model, inputs)

    def _decode(self, tokens):
        tokenizer = self.eval_preprocessor.tokenizer
        return tokenizer.detokenize(tokens.tolist())

    def evaluation_step(self, data):
        model = self.model.module if self._dist else self.model
        model.eval()

        if self._is_pair(data):
            return self._generate_eval(model, data)
        else:
            return self._forward_eval(model, data)

    @staticmethod
    def _is_pair(data: Dict[str, Any]) -> bool:
        return 'is_pair' in data and bool(data['is_pair'][0])

    def _generate_eval(self, model: GPT3ForTextGeneration,
                       data: Dict[str, Any]) -> Dict[str, Any]:
        data['do_sample'] = False
        result = model.generate(data)

        prompt_length: List[int] = data['prompt_length']
        result['preds'] = [
            self._decode(seq[skip_len:])
            for seq, skip_len in zip(result['sequences'], prompt_length)
        ]
        data['tgts'] = [
            self._decode(seq[skip_len - 1:])
            for seq, skip_len in zip(data['labels'], prompt_length)
        ]
        return result

    def _forward_eval(self, model: GPT3ForTextGeneration,
                      data: Dict[str, Any]) -> Dict[str, Any]:
        return model.forward(data)
