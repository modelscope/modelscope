# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Any, Dict, List

from modelscope.metainfo import Trainers
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

    def _decode(self, tokens):
        tokenizer = self.eval_preprocessor.tokenizer
        return tokenizer.detokenize(tokens.tolist())

    def evaluation_step(self, data):
        model = self.model.module if self._dist else self.model
        model.eval()

        if 'inputs_len' in data:
            return self._generate_eval(model, data)
        else:
            return self._forward_eval(model, data)

    def _generate_eval(self, model: GPT3ForTextGeneration,
                       data: Dict[str, Any]) -> Dict[str, Any]:
        # Force greedy decoding in non-open tasks
        data.update(top_k=1, top_p=0.)
        result = model.generate(data)

        prompts_len: List[int] = data['prompts_len']
        result['preds'] = [
            self._decode(seq[skip_len:])
            for seq, skip_len in zip(result['sequences'], prompts_len)
        ]
        data['tgts'] = [
            self._decode(seq[skip_len - 1:])
            for seq, skip_len in zip(data['labels'], prompts_len)
        ]
        return result

    def _forward_eval(self, model: GPT3ForTextGeneration,
                      data: Dict[str, Any]) -> Dict[str, Any]:
        return model.forward(data)
