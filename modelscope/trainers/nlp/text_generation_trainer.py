# Copyright (c) Alibaba, Inc. and its affiliates.

from collections.abc import Mapping

import torch

from modelscope.metainfo import Trainers
from modelscope.trainers import NlpEpochBasedTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.file_utils import func_receive_dict_inputs


@TRAINERS.register_module(module_name=Trainers.text_generation_trainer)
class TextGenerationTrainer(NlpEpochBasedTrainer):

    def _decode(self, tokens):
        return self.eval_preprocessor.decode(
            tokens.tolist(), skip_special_tokens=True)

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

        result['preds'] = [self._decode(seq) for seq in result['sequences']]
        data['tgts'] = [self._decode(seq) for seq in data['labels']]
        assert len(result['preds']) == len(data['tgts'])

        return result
