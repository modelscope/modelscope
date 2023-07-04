# Copyright (c) Alibaba, Inc. and its affiliates.
import collections
import io
import os
import sys
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
from modelscope.utils.constant import ModelFile, Tasks

HOP_LENGTH = 960
N_FFT = 1920
WINDOW_NAME_HAM = 'hamming'
STFT_WIN_LEN = 1920
WINLEN = 3840
STRIDE = 1920


@PIPELINES.register_module(
    Tasks.acoustic_noise_suppression,
    module_name=Pipelines.speech_dfsmn_ans_psm_48k_causal)
class ANSDFSMNPipeline(Pipeline):
    """ANS (Acoustic Noise Suppression) inference pipeline based on DFSMN model.

    Args:
        stream_mode: set its work mode, default False
        In stream model, it accepts bytes as pipeline input that should be the audio data in PCM format.
        In normal model, it accepts str and treat it as the path of local wav file or the http link of remote wav file.
    """
    SAMPLE_RATE = 48000

    def __init__(self, model, **kwargs):
        super().__init__(model=model, **kwargs)
        model_bin_file = os.path.join(self.model.model_dir,
                                      ModelFile.TORCH_MODEL_BIN_FILE)
        if os.path.exists(model_bin_file):
            checkpoint = torch.load(model_bin_file, map_location=self.device)
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.stream_mode = kwargs.get('stream_mode', False)
        if self.stream_mode:
            # the unit of WINLEN and STRIDE is frame, 1 frame of 16bit = 2 bytes
            byte_buffer_length = \
                (WINLEN + STRIDE * (self.model.lorder - 1)) * 2
            self.buffer = collections.deque(maxlen=byte_buffer_length)
            # padding head
            for i in range(STRIDE * 2):
                self.buffer.append(b'\0')
            # it processes WINLEN frames at the first time, then STRIDE frames
            self.byte_length_remain = (STRIDE * 2 - WINLEN) * 2
            self.first_forward = True
            self.tensor_give_up_length = (WINLEN - STRIDE) // 2

        window = torch.hamming_window(
            STFT_WIN_LEN, periodic=False, device=self.device)

        def stft(x):
            return torch.stft(
                x,
                N_FFT,
                HOP_LENGTH,
                STFT_WIN_LEN,
                center=False,
                window=window,
                return_complex=False)

        def istft(x, slen):
            return librosa.istft(
                x,
                hop_length=HOP_LENGTH,
                win_length=STFT_WIN_LEN,
                window=WINDOW_NAME_HAM,
                center=False,
                length=slen)

        self.stft = stft
        self.istft = istft

    def preprocess(self, inputs: Input, **preprocess_params) -> Dict[str, Any]:
        if self.stream_mode:
            if not isinstance(inputs, bytes):
                raise TypeError('Only support bytes in stream mode.')
            if len(inputs) > self.buffer.maxlen:
                raise ValueError(
                    f'inputs length too large: {len(inputs)} > {self.buffer.maxlen}'
                )
            tensor_list = []
            current_index = 0
            while self.byte_length_remain + len(
                    inputs) - current_index >= STRIDE * 2:
                byte_length_to_add = STRIDE * 2 - self.byte_length_remain
                for i in range(current_index,
                               current_index + byte_length_to_add):
                    self.buffer.append(inputs[i].to_bytes(
                        1, byteorder=sys.byteorder, signed=False))
                bytes_io = io.BytesIO()
                for b in self.buffer:
                    bytes_io.write(b)
                data = np.frombuffer(bytes_io.getbuffer(), dtype=np.int16)
                data_tensor = torch.from_numpy(data).type(torch.FloatTensor)
                tensor_list.append(data_tensor)
                self.byte_length_remain = 0
                current_index += byte_length_to_add
            for i in range(current_index, len(inputs)):
                self.buffer.append(inputs[i].to_bytes(
                    1, byteorder=sys.byteorder, signed=False))
                self.byte_length_remain += 1
            return {'audio': tensor_list}
        else:
            if isinstance(inputs, str):
                data_bytes = File.read(inputs)
            elif isinstance(inputs, bytes):
                data_bytes = inputs
            else:
                raise TypeError(f'Unsupported type {type(inputs)}.')
            data_tensor = self.bytes2tensor(data_bytes)
            return {'audio': data_tensor}

    def bytes2tensor(self, file_bytes):
        data1, fs = sf.read(io.BytesIO(file_bytes))
        data1 = data1.astype(np.float32)
        if len(data1.shape) > 1:
            data1 = data1[:, 0]
        if fs != self.SAMPLE_RATE:
            data1 = librosa.resample(data1, fs, self.SAMPLE_RATE)
        data = data1 * 32768
        data_tensor = torch.from_numpy(data).type(torch.FloatTensor)
        return data_tensor

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        if self.stream_mode:
            bytes_io = io.BytesIO()
            for origin_audio in inputs['audio']:
                masked_sig = self._forward(origin_audio)
                if self.first_forward:
                    masked_sig = masked_sig[:-self.tensor_give_up_length]
                    self.first_forward = False
                else:
                    masked_sig = masked_sig[-WINLEN:]
                    masked_sig = masked_sig[self.tensor_give_up_length:-self.
                                            tensor_give_up_length]
                bytes_io.write(masked_sig.astype(np.int16).tobytes())
            outputs = bytes_io.getvalue()
        else:
            origin_audio = inputs['audio']
            masked_sig = self._forward(origin_audio)
            outputs = masked_sig.astype(np.int16).tobytes()
        return {OutputKeys.OUTPUT_PCM: outputs}

    def _forward(self, origin_audio):
        with torch.no_grad():
            audio_in = origin_audio.unsqueeze(0)
            import torchaudio
            fbanks = torchaudio.compliance.kaldi.fbank(
                audio_in,
                dither=1.0,
                frame_length=40.0,
                frame_shift=20.0,
                num_mel_bins=120,
                sample_frequency=self.SAMPLE_RATE,
                window_type=WINDOW_NAME_HAM)
            fbanks = fbanks.unsqueeze(0)
            masks = self.model(fbanks)
            spectrum = self.stft(origin_audio)
            masks = masks.permute(2, 1, 0)
            masked_spec = (spectrum * masks).cpu()
        masked_spec = masked_spec.detach().numpy()
        masked_spec_complex = masked_spec[:, :, 0] + 1j * masked_spec[:, :, 1]
        masked_sig = self.istft(masked_spec_complex, len(origin_audio))
        return masked_sig

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if not self.stream_mode and 'output_path' in kwargs.keys():
            sf.write(
                kwargs['output_path'],
                np.frombuffer(inputs[OutputKeys.OUTPUT_PCM], dtype=np.int16),
                self.SAMPLE_RATE)
        return inputs
