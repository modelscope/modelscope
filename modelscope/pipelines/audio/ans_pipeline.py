# Copyright (c) Alibaba, Inc. and its affiliates.

import io
from typing import Any, Dict

import librosa
import numpy as np
import soundfile as sf
import torch

from modelscope.fileio import File
from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.audio.audio_utils import audio_norm
from modelscope.utils.constant import Tasks


@PIPELINES.register_module(
    Tasks.acoustic_noise_suppression,
    module_name=Pipelines.speech_frcrn_ans_cirm_16k)
class ANSPipeline(Pipeline):
    r"""ANS (Acoustic Noise Suppression) Inference Pipeline .

    When invoke the class with pipeline.__call__(), it accept only one parameter:
        inputs(str): the path of wav file
    """
    SAMPLE_RATE = 16000

    def __init__(self, model, **kwargs):
        """
        use `model` and `preprocessor` to create a kws pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        self.model.eval()

    def preprocess(self, inputs: Input, **preprocess_params) -> Dict[str, Any]:
        if isinstance(inputs, bytes):
            data1, fs = sf.read(io.BytesIO(inputs))
        elif isinstance(inputs, str):
            file_bytes = File.read(inputs)
            data1, fs = sf.read(io.BytesIO(file_bytes))
        else:
            raise TypeError(f'Unsupported type {type(inputs)}.')
        if len(data1.shape) > 1:
            data1 = data1[:, 0]
        if fs != self.SAMPLE_RATE:
            data1 = librosa.resample(data1, fs, self.SAMPLE_RATE)
        data1 = audio_norm(data1)
        data = data1.astype(np.float32)
        inputs = np.reshape(data, [1, data.shape[0]])
        return {'ndarray': inputs, 'nsamples': data.shape[0]}

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        ndarray = inputs['ndarray']
        if isinstance(ndarray, torch.Tensor):
            ndarray = ndarray.cpu().numpy()
        nsamples = inputs['nsamples']
        decode_do_segement = False
        window = 16000
        stride = int(window * 0.75)
        print('inputs:{}'.format(ndarray.shape))
        b, t = ndarray.shape  # size()
        if t > window * 120:
            decode_do_segement = True

        if t < window:
            ndarray = np.concatenate(
                [ndarray, np.zeros((ndarray.shape[0], window - t))], 1)
        elif t < window + stride:
            padding = window + stride - t
            print('padding: {}'.format(padding))
            ndarray = np.concatenate(
                [ndarray, np.zeros((ndarray.shape[0], padding))], 1)
        else:
            if (t - window) % stride != 0:
                padding = t - (t - window) // stride * stride
                print('padding: {}'.format(padding))
                ndarray = np.concatenate(
                    [ndarray, np.zeros((ndarray.shape[0], padding))], 1)
        print('inputs after padding:{}'.format(ndarray.shape))
        with torch.no_grad():
            ndarray = torch.from_numpy(np.float32(ndarray)).to(self.device)
            b, t = ndarray.shape
            if decode_do_segement:
                outputs = np.zeros(t)
                give_up_length = (window - stride) // 2
                current_idx = 0
                while current_idx + window <= t:
                    print('current_idx: {}'.format(current_idx))
                    tmp_input = dict(noisy=ndarray[:, current_idx:current_idx
                                                   + window])
                    tmp_output = self.model(
                        tmp_input, )['wav_l2'][0].cpu().numpy()
                    end_index = current_idx + window - give_up_length
                    if current_idx == 0:
                        outputs[current_idx:
                                end_index] = tmp_output[:-give_up_length]
                    else:
                        outputs[current_idx
                                + give_up_length:end_index] = tmp_output[
                                    give_up_length:-give_up_length]
                    current_idx += stride
            else:
                outputs = self.model(
                    dict(noisy=ndarray))['wav_l2'][0].cpu().numpy()
        outputs = (outputs[:nsamples] * 32768).astype(np.int16).tobytes()
        return {OutputKeys.OUTPUT_PCM: outputs}

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if 'output_path' in kwargs.keys():
            sf.write(
                kwargs['output_path'],
                np.frombuffer(inputs[OutputKeys.OUTPUT_PCM], dtype=np.int16),
                self.SAMPLE_RATE)
        return inputs
