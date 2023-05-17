# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Union

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.nlp import DocumentGroundedDialogGenerateModel
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import DocumentGroundedDialogGeneratePreprocessor
from modelscope.utils.constant import Tasks

__all__ = ['DocumentGroundedDialogGeneratePipeline']


@PIPELINES.register_module(
    Tasks.document_grounded_dialog_generate,
    module_name=Pipelines.document_grounded_dialog_generate)
class DocumentGroundedDialogGeneratePipeline(Pipeline):

    def __init__(
            self,
            model: Union[DocumentGroundedDialogGenerateModel, str],
            preprocessor: DocumentGroundedDialogGeneratePreprocessor = None,
            config_file: str = None,
            device: str = 'gpu',
            auto_collate=True,
            **kwargs):
        """The Generate pipeline for document grounded dialog

        Args:
            model: A model instance or a model local dir or a model id in the model hub.
            preprocessor: A preprocessor instance.
            config_file: Path to config file.
            device: Device to run the model.
            auto_collate: Apply auto collate.
            **kwargs: The preprocessor kwargs passed into the preprocessor's constructor.

        Examples:
            >>> from modelscope.pipelines import pipeline
            >>> pipe_ins = pipeline('document-grounded-dialog-generate', model='damo/nlp_convai_generate')
        """
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate,
            compile=kwargs.pop('compile', False),
            compile_options=kwargs.pop('compile_options', {}))

        if preprocessor is None:
            self.preprocessor = DocumentGroundedDialogGeneratePreprocessor(
                self.model.model_dir, **kwargs)

    def forward(self, inputs: Union[list, Dict[str, Any]],
                **forward_params) -> Dict[str, Any]:
        return {'generated_ids': self.model.generate(inputs)}

    def postprocess(self, inputs: Union[list, Dict[str, Any]],
                    **postprocess_params) -> Dict[str, Any]:
        predictions = self.preprocessor.generation_tokenizer.batch_decode(
            inputs['generated_ids'],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        return {OutputKeys.TEXT: predictions}

    def _collate_fn(self, data):
        return data
