# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from collections.abc import Mapping
from typing import List

import torch
from megatron_util import mpu

from modelscope.metainfo import Trainers
from modelscope.models import TorchModel
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.nlp_trainer import NlpEpochBasedTrainer
from modelscope.utils.config import Config
from modelscope.utils.file_utils import func_receive_dict_inputs


@TRAINERS.register_module(module_name=Trainers.gpt_moe_trainer)
class GPTMoETrainer(NlpEpochBasedTrainer):

    def rebuild_config(self, cfg: Config):
        super().rebuild_config(cfg)
        cfg.model.rank = int(os.environ.get('LOCAL_RANK', -1))
        cfg.model.master_ip = os.environ.get('MASTER_ADDR', '127.0.0.1')
        cfg.model.master_port = os.environ.get('MASTER_PORT', '29500')
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

        with torch.no_grad():
            if isinstance(
                    data,
                    Mapping) and not func_receive_dict_inputs(model.generate):
                result = model.generate(**data)
            else:
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
        assert len(result['preds']) == len(data['tgts'])

        return result
