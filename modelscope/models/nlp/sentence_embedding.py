# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

import numpy as np

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.nlp.structbert import SbertPreTrainedModel
from modelscope.utils.constant import Tasks

__all__ = ['SentenceEmbedding']


@MODELS.register_module(Tasks.sentence_embedding, module_name=Models.bert)
class SentenceEmbedding(TorchModel, SbertPreTrainedModel):
    base_model_prefix: str = 'bert'
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r'position_ids']

    def __init__(self, config, model_dir):
        super().__init__(model_dir)
        self.config = config
        setattr(self, self.base_model_prefix, self.build_base_model())

    def build_base_model(self):
        from .structbert import SbertModel
        return SbertModel(self.config, add_pooling_layer=False)

    def forward(self, input: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """return the result by the model

        Args:
            input (Dict[str, Any]): the preprocessed data

        Returns:
            Dict[str, np.ndarray]: results
                Example:
                    {
                        'predictions': array([1]), # lable 0-negative 1-positive
                        'probabilities': array([[0.11491239, 0.8850876 ]], dtype=float32),
                        'logits': array([[-0.53860897,  1.5029076 ]], dtype=float32) # true value
                    }
        """
        return self.base_model(**input)

    def postprocess(self, inputs: Dict[str, np.ndarray],
                    **kwargs) -> Dict[str, np.ndarray]:
        embs = inputs['last_hidden_state'][:, 0].cpu().numpy()
        num_sent = embs.shape[0]
        if num_sent >= 2:
            scores = np.dot(embs[0:1, ], np.transpose(embs[1:, ],
                                                      (1, 0))).tolist()[0]
        else:
            scores = []
        result = {'text_embedding': embs, 'scores': scores}

        return result

    @classmethod
    def _instantiate(cls, **kwargs):
        """Instantiate the model.

        @param kwargs: Input args.
                    model_dir: The model dir used to load the checkpoint and the label information.
        @return: The loaded model, which is initialized by transformers.PreTrainedModel.from_pretrained
        """
        model_args = {}

        return super(SbertPreTrainedModel, SentenceEmbedding).from_pretrained(
            pretrained_model_name_or_path=kwargs.get('model_dir'),
            model_dir=kwargs.get('model_dir'),
            **model_args)
