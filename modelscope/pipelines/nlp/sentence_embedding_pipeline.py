# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Optional, Union

import numpy as np

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import Tasks

__all__ = ['SentenceEmbeddingPipeline']


@PIPELINES.register_module(
    Tasks.sentence_embedding, module_name=Pipelines.sentence_embedding)
class SentenceEmbeddingPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 first_sequence='first_sequence',
                 **kwargs):
        """Use `model` and `preprocessor` to create a nlp text dual encoder then generates the text representation.
        Args:
            model (str or Model): Supply either a local model dir which supported the WS task,
            or a model id from the model hub, or a torch model instance.
            preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for
            the model if supplied.
            sequence_length: Max sequence length in the user's custom scenario. 128 will be used as a default value.
        """
        model = Model.from_pretrained(model) if isinstance(model,
                                                           str) else model
        if preprocessor is None:
            preprocessor = Preprocessor.from_pretrained(
                model.model_dir if isinstance(model, Model) else model,
                first_sequence=first_sequence,
                sequence_length=kwargs.pop('sequence_length', 128))
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        return self.model(**inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): _description_

        Returns:
            Dict[str, Any]: the predicted text representation
        """
        embs = inputs['last_hidden_state'][:, 0].cpu().numpy()
        num_sent = embs.shape[0]
        if num_sent >= 2:
            scores = np.dot(embs[0:1, ], np.transpose(embs[1:, ],
                                                      (1, 0))).tolist()[0]
        else:
            scores = []
        return {OutputKeys.TEXT_EMBEDDING: embs, OutputKeys.SCORES: scores}
