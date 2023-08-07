# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Union

import numpy as np

from modelscope.metainfo import Pipelines
from modelscope.models.base.base_model import Model
from modelscope.outputs import OutputKeys, TextClassificationModelOutput
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.speaker_diarization_dialogue_detection,
    module_name=Pipelines.speaker_diarization_dialogue_detection)
class SpeakerDiarizationDialogueDetectionPipeline(Pipeline):
    r"""The inference pipeline for Speaker Diarization Dialogue Detection Task.
        Dialogue Detection Task is to detect whether the text transcribed from ASR module is a dialogue from different
        speakers, Speaker Diarization Task have been proved to benefit from Dialogue Detection Task.
        The input of the Dialogue Detection is Text, and the output is the score and labels.
        The labels is always in ['dialogue', 'non_dialogue'] in our pipeline.


        Examples:
            >>> from modelscope.pipelines import pipeline
            >>> pipeline_ins = pipeline('speaker_diarization_dialogue_detection',
                                        model='damo/speech_bert_dialogue-detetction_speaker-diarization_chinese')
            >>> input_text = "侦探小说从19世纪中期开始发展。美国作家埃德加‧爱伦‧坡被认为是西方侦探小说的鼻祖。"
            >>> print(pipeline_ins(input_text))
    """

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Preprocessor = None,
                 config_file: str = None,
                 device: str = 'gpu',
                 auto_collate=True,
                 **kwargs):
        r"""Create the pipeline for speaker diarization dialogue detection inference.

            Args:
                model (`str` or `Model` or Module instance): A model instance or a model local dir or a model id
                                                             in model hub.
                preprocessor: (`Preprocessor`, `optional`): A Preprocessor instance usually for tokenizer.
                kwargs (dict, `optional`): Extra kwargs, some of args will pass into preprocessor's constructor.
        """
        super(SpeakerDiarizationDialogueDetectionPipeline, self).__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate,
            compile=kwargs.pop('compile', False),
            compile_options=kwargs.pop('compile_options', {}))
        assert isinstance(self.model, Model), \
            f'Please check whether the model {model} and config {config} exists in {ModelFile.CONFIGURATION}'

        if preprocessor is None:
            first_sequence = kwargs.pop('first_sequence', 'text')
            second_sequence = kwargs.pop('second_sequence', None)
            sequence_length = kwargs.pop('sequence_length', 512)
            self.preprocessor = Preprocessor.from_pretrained(
                self.model.model_dir, **{
                    'first_sequence': first_sequence,
                    'second_sequence': second_sequence,
                    'sequence_length': sequence_length,
                    **kwargs
                })

        if hasattr(self.preprocessor, 'id2label'):
            self.id2label = self.preprocessor.id2label

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        return self.model(**inputs, **forward_params)

    def postprocess(self,
                    inputs: Union[Dict[str, Any],
                                  TextClassificationModelOutput],
                    topk: int = None) -> Dict[str, Any]:
        r"""The postprocess is to align the model output and the final outputs.

            Args:
                inputs (`Dict[str, Any]` or `TextClassificationModelOutput`): The model output, you can check
                    `TextClassificationModelOutput` for more details.
                topk(int): The topk probs to take
            Returns:
                Dict[str, Any]: The final model prediction results.
                    scores: The probabilities of each label.
                    labels: The real labels read from id2label: ['dialogue', 'non_dialogue']
        """
        if getattr(self, 'id2label', None) is None:
            logger.warning(
                'The id2label mapping is None, will return original ids.')
        logits = inputs[OutputKeys.LOGITS].detach().cpu().numpy()
        if logits.shape[0] == 1:
            logits = logits[0]

        def softmax(logits):
            exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            return exp / exp.sum(axis=-1, keepdims=True)

        probs = softmax(logits)
        num_classes = probs.shape[-1]
        topk = min(topk, num_classes) if topk is not None else num_classes
        top_indices = np.argpartition(probs, -topk)[-topk:]
        probs = np.take_along_axis(probs, top_indices, axis=-1).tolist()

        def map_to_label(id):
            if getattr(self, 'id2label', None) is not None:
                if id in self.id2label:
                    return self.id2label[id]
                elif str(id) in self.id2label:
                    return self.id2label[str(id)]
                else:
                    raise Exception(
                        f'id {id} not found in id2label: {self.id2label}')
            else:
                return id

        v_func = np.vectorize(map_to_label)
        top_indices = v_func(top_indices).tolist()
        probs = list(reversed(probs))
        top_indices = list(reversed(top_indices))
        return {OutputKeys.SCORES: probs, OutputKeys.LABELS: top_indices}
