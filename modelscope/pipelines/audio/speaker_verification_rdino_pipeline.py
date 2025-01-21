# Copyright (c) Alibaba, Inc. and its affiliates.

import io
from typing import Any, Dict, List, Union

import soundfile as sf
import torch

from modelscope.fileio import File
from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import InputModel, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.speaker_verification,
    module_name=Pipelines.speaker_verification_rdino)
class RDINO_Pipeline(Pipeline):
    """Speaker Verification Inference Pipeline
    use `model` to create a Speaker Verification pipeline.

    Args:
        model (SpeakerVerificationPipeline): A model instance, or a model local dir, or a model id in the model hub.
        kwargs (dict, `optional`):
            Extra kwargs passed into the pipeline's constructor.
    Example:
    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.utils.constant import Tasks
    >>> p = pipeline(
    >>>    task=Tasks.speaker_verification, model='damo/speech_ecapa-tdnn_sv_en_voxceleb_16k')
    >>> print(p([audio_1, audio_2]))

    """

    def __init__(self, model: InputModel, **kwargs):
        """use `model` to create a speaker verification pipeline for prediction
        Args:
            model (str): a valid offical model id
        """
        super().__init__(model=model, **kwargs)
        self.model_config = self.model.model_config
        self.config = self.model.other_config
        self.thr = self.config['yesOrno_thr']

    def __call__(self,
                 in_audios: List[str],
                 thr: float = None) -> Dict[str, Any]:
        if thr is not None:
            self.thr = thr
        if self.thr < -1 or self.thr > 1:
            raise ValueError(
                'modelscope error: the thr value should be in [-1, 1], but found to be %f.'
                % self.thr)
        outputs = self.preprocess(in_audios)
        outputs = self.forward(outputs)
        outputs = self.postprocess(outputs)

        return outputs

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        emb1 = self.model(inputs['data1'])
        emb2 = self.model(inputs['data2'])

        return {'emb1': emb1, 'emb2': emb2}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        score = self.compute_cos_similarity(inputs['emb1'], inputs['emb2'])
        score = round(score, 5)
        if score >= self.thr:
            ans = 'yes'
        else:
            ans = 'no'

        return {OutputKeys.SCORE: score, OutputKeys.TEXT: ans}

    def preprocess(self, inputs: List[str],
                   **preprocess_params) -> Dict[str, Any]:
        if len(inputs) != 2:
            raise ValueError(
                'modelscope error: Two input audio files are required.')
        output = {}
        for i in range(len(inputs)):
            if isinstance(inputs[i], str):
                file_bytes = File.read(inputs[i])
                data, fs = sf.read(io.BytesIO(file_bytes), dtype='float32')
                if len(data.shape) == 2:
                    data = data[:, 0]
                if fs != self.model_config['sample_rate']:
                    raise ValueError(
                        'modelscope error: Only support %d sample rate files'
                        % self.model_cfg['sample_rate'])
                output['data%d' %
                       (i + 1)] = torch.from_numpy(data).unsqueeze(0)
            else:
                raise ValueError(
                    'modelscope error: The input type is temporarily restricted to audio file address'
                    % i)
        return output

    def compute_cos_similarity(self, emb1: torch.Tensor,
                               emb2: torch.Tensor) -> float:
        assert len(emb1.shape) == 2 and len(emb2.shape) == 2
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cosine = cos(emb1, emb2)
        return cosine.item()
