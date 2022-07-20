from typing import Any, Dict

import torch.cuda

from modelscope.metainfo import Models
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks
from ..base import Model
from ..builder import MODELS
from .ofa import OFAModel, OFATokenizer
from .ofa.generate import sequence_generator as sg
from .ofa.generate.utils import move_to_device

__all__ = ['OfaForImageCaptioning']


@MODELS.register_module(Tasks.image_captioning, module_name=Models.ofa)
class OfaForImageCaptioning(Model):

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir=model_dir, *args, **kwargs)
        model = OFAModel.from_pretrained(model_dir)

        self.model = model.module if hasattr(model, 'module') else model
        self.tokenizer = OFATokenizer.from_pretrained(model_dir)
        self.tokenizer.add_tokens(['<code_{}>'.format(i) for i in range(8192)])
        self.tokenizer.add_tokens(['<bin_{}>'.format(i) for i in range(1000)])
        self._device = torch.device('cuda') if torch.cuda.is_available() \
            else torch.device('cpu')
        self.model.to(self._device)
        # Initialize generator
        sg_args = {
            'tokenizer': self.tokenizer,
            'beam_size': 5,
            'max_len_b': 16,
            'min_len': 1,
            'no_repeat_ngram_size': 3,
            'constraint_range': None
        }
        if hasattr(kwargs, 'beam_search'):
            sg_args.update(kwargs['beam_search'])
        self.generator = sg.SequenceGenerator(**sg_args)

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        input = move_to_device(input, self._device)
        gen_output = self.generator.generate([self.model], input)
        gen = [gen_output[i][0]['tokens'] for i in range(len(gen_output))]
        result = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        return {'image_id': '42', OutputKeys.CAPTION: result[0]}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # What should we do here ?
        return inputs
