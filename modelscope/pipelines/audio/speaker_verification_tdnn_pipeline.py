# Copyright (c) Alibaba, Inc. and its affiliates.

import io
import os
from typing import Any, Dict, List, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

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
    module_name=Pipelines.speaker_verification_tdnn)
class SpeakerVerificationTDNNPipeline(Pipeline):
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
        self.save_dict = {}

    def __call__(self,
                 in_audios: Union[np.ndarray, list],
                 save_dir: str = None,
                 output_emb: bool = False,
                 thr: float = None):
        if thr is not None:
            self.thr = thr
        if self.thr < -1 or self.thr > 1:
            raise ValueError(
                'modelscope error: the thr value should be in [-1, 1], but found to be %f.'
                % self.thr)
        wavs = self.preprocess(in_audios)
        embs = self.forward(wavs)
        outputs = self.postprocess(embs, in_audios, save_dir)
        if output_emb:
            self.save_dict['outputs'] = outputs
            self.save_dict['embs'] = embs.numpy()
            return self.save_dict
        else:
            return outputs

    def forward(self, inputs: list):
        embs = []
        for x in inputs:
            embs.append(self.model(x))
        embs = torch.cat(embs)
        return embs

    def postprocess(self,
                    inputs: torch.Tensor,
                    in_audios: Union[np.ndarray, list],
                    save_dir=None):
        if isinstance(in_audios[0], str) and save_dir is not None:
            # save the embeddings
            os.makedirs(save_dir, exist_ok=True)
            for i, p in enumerate(in_audios):
                save_path = os.path.join(
                    save_dir, '%s.npy' %
                    (os.path.basename(p).rsplit('.', 1)[0]))
                np.save(save_path, inputs[i].numpy())

        if len(inputs) == 2:
            # compute the score
            score = self.compute_cos_similarity(inputs[0], inputs[1])
            score = round(score, 5)
            if score >= self.thr:
                ans = 'yes'
            else:
                ans = 'no'
            output = {OutputKeys.SCORE: score, OutputKeys.TEXT: ans}
        else:
            output = {OutputKeys.TEXT: 'No similarity score output'}

        return output

    def preprocess(self, inputs: Union[np.ndarray, list]):
        output = []
        for i in range(len(inputs)):
            if isinstance(inputs[i], str):
                file_bytes = File.read(inputs[i])
                data, fs = sf.read(io.BytesIO(file_bytes), dtype='float32')
                if len(data.shape) == 2:
                    data = data[:, 0]
                data = torch.from_numpy(data).unsqueeze(0)
                if fs != self.model_config['sample_rate']:
                    logger.warning(
                        'The sample rate of audio is not %d, resample it.'
                        % self.model_config['sample_rate'])
                    data, fs = torchaudio.sox_effects.apply_effects_tensor(
                        data,
                        fs,
                        effects=[[
                            'rate',
                            str(self.model_config['sample_rate'])
                        ]])
                data = data.squeeze(0)
            elif isinstance(inputs[i], np.ndarray):
                assert len(
                    inputs[i].shape
                ) == 1, 'modelscope error: Input array should be [N, T]'
                data = inputs[i]
                if data.dtype in ['int16', 'int32', 'int64']:
                    data = (data / (1 << 15)).astype('float32')
                else:
                    data = data.astype('float32')
                data = torch.from_numpy(data)
            else:
                raise ValueError(
                    'modelscope error: The input type is restricted to audio address and nump array.'
                )
            output.append(data)
        return output

    def compute_cos_similarity(self, emb1: Union[np.ndarray, torch.Tensor],
                               emb2: Union[np.ndarray, torch.Tensor]) -> float:
        if isinstance(emb1, np.ndarray):
            emb1 = torch.from_numpy(emb1)
        if isinstance(emb2, np.ndarray):
            emb2 = torch.from_numpy(emb2)
        if len(emb1.shape):
            emb1 = emb1.unsqueeze(0)
        if len(emb2.shape):
            emb2 = emb2.unsqueeze(0)
        assert len(emb1.shape) == 2 and len(emb2.shape) == 2
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cosine = cos(emb1, emb2)
        return cosine.item()
