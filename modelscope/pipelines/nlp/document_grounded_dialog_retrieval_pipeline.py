# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path
from typing import Any, Dict, List, Union

import faiss
import json
import numpy as np

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.nlp import DocumentGroundedDialogRetrievalModel
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import \
    DocumentGroundedDialogRetrievalPreprocessor
from modelscope.utils.constant import ModeKeys, Tasks

__all__ = ['DocumentGroundedDialogRetrievalPipeline']


@PIPELINES.register_module(
    Tasks.document_grounded_dialog_retrieval,
    module_name=Pipelines.document_grounded_dialog_retrieval)
class DocumentGroundedDialogRetrievalPipeline(Pipeline):

    def __init__(
            self,
            model: Union[DocumentGroundedDialogRetrievalModel, str],
            preprocessor: DocumentGroundedDialogRetrievalPreprocessor = None,
            config_file: str = None,
            device: str = 'gpu',
            auto_collate=True,
            index_path: str = None,
            per_gpu_batch_size: int = 32,
            **kwargs):
        """The Retrieval pipeline for document grounded dialog.
        Args:
            model: A model instance or a model local dir or a model id in the model hub.
            preprocessor: A preprocessor instance.
            config_file: Path to config file.
            device: Device to run the model.
            auto_collate: Apply auto collate.
            index_path: Index file path.
            per_gpu_batch_size: Batch size per GPU to run the code.
            **kwargs: The preprocessor kwargs passed into the preprocessor's constructor.

        Examples:
            >>> from modelscope.pipelines import pipeline
            >>> pipe_ins = pipeline('document-grounded-dialog-retrieval', model='damo/nlp_convai_retrieval')

        """
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate)

        if preprocessor is None:
            self.preprocessor = DocumentGroundedDialogRetrievalPreprocessor(
                self.model.model_dir, **kwargs)
        self.per_gpu_batch_size = per_gpu_batch_size
        self.passages_index = []
        self.passages = []
        self.index = None
        self.load_index(index_path)

    def forward(self, inputs: Union[list, Dict[str, Any]],
                **forward_params) -> Dict[str, Any]:
        query_vector = self.model.encode_query(
            inputs).detach().cpu().numpy().astype('float32')
        D, Index = self.index.search(query_vector, 20)
        return {'retrieved_ids': Index.tolist()}

    def postprocess(self, inputs: Union[list, Dict[str, Any]],
                    **postprocess_params) -> Dict[str, Any]:
        predictions = [[self.passages[x] for x in retrieved_ids]
                       for retrieved_ids in inputs['retrieved_ids']]
        return {OutputKeys.OUTPUT: predictions}

    def _collate_fn(self, data):
        return data

    def load_index(self, index_path: str = None):
        if not index_path:
            index_path = os.path.join(self.model.model_dir,
                                      'passages_index.json')
        with open(index_path) as f:
            passage_index = json.load(f)
        self.passages_index = passage_index
        self.passages = [x['passage'] for x in passage_index]
        all_ctx_vector = np.array([x['vector']
                                   for x in passage_index]).astype('float32')
        index = faiss.IndexFlatIP(all_ctx_vector.shape[-1])
        index.add(all_ctx_vector)
        self.index = index

    def save_index(self, index_path: str = None):
        if not index_path:
            index_path = os.path.join(self.model.model_dir,
                                      'passages_index.json')
        with open(index_path, 'w') as f:
            json.dump(self.passage_index, f, ensure_ascii=False, indent=4)

    def add_passage(self, passages: List[str]):
        all_ctx_vector = []
        for mini_batch in range(0, len(passages), self.per_gpu_batch_size):
            context = passages[mini_batch:mini_batch + self.per_gpu_batch_size]
            processed = self.preprocessor({'context': context},
                                          invoke_mode=ModeKeys.INFERENCE,
                                          input_type='context')
            sub_ctx_vector = self.model.encode_context(
                processed).detach().cpu().numpy()
            all_ctx_vector.append(sub_ctx_vector)
        all_ctx_vector = np.concatenate(all_ctx_vector, axis=0)
        all_ctx_vector = np.array(all_ctx_vector).astype('float32')
        for passage, vector in zip(passages, all_ctx_vector):
            self.passages_index.append({
                'passage': passage,
                'vector': vector.tolist()
            })
        self.passages = [x['passage'] for x in self.passage_index]
        all_ctx_vector = np.array([x['vector'] for x in self.passage_index
                                   ]).astype('float32')
        index = faiss.IndexFlatIP(all_ctx_vector.shape[-1])
        index.add(all_ctx_vector)
        self.index = index
