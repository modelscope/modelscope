# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import ModelFile, Tasks

__all__ = ['SentenceEmbeddingPipeline']


@PIPELINES.register_module(
    Tasks.sentence_embedding, module_name=Pipelines.sentence_embedding)
class SentenceEmbeddingPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 config_file: str = None,
                 device: str = 'gpu',
                 auto_collate=True,
                 sequence_length=128,
                 **kwargs):
        """Use `model` and `preprocessor` to create a nlp text dual encoder then generates the text representation.
        Args:
            model (str or Model): Supply either a local model dir which supported the WS task,
            or a model id from the model hub, or a torch model instance.
            preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for
            the model if supplied.
            kwargs (dict, `optional`):
                Extra kwargs passed into the preprocessor's constructor.
        """
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate)

        assert isinstance(self.model, Model), \
            f'please check whether model config exists in {ModelFile.CONFIGURATION}'

        if preprocessor is None:
            self.preprocessor = Preprocessor.from_pretrained(
                self.model.model_dir,
                sequence_length=sequence_length,
                **kwargs)

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
        embeddings = inputs['query_embeddings']
        doc_embeddings = inputs['doc_embeddings']
        if doc_embeddings is not None:
            embeddings = torch.cat((embeddings, doc_embeddings), dim=0)
        embeddings = embeddings.detach().cpu().numpy()
        if doc_embeddings is not None:
            scores = np.dot(embeddings[0:1, ],
                            np.transpose(embeddings[1:, ], (1, 0))).tolist()[0]
        else:
            scores = []
        return {
            OutputKeys.TEXT_EMBEDDING: embeddings,
            OutputKeys.SCORES: scores
        }
