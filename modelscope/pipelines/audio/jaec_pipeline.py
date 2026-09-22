# Copyright (c) Alibaba, Inc. and its affiliates.

import io
import os
from typing import Any, Dict, Tuple

import numpy as np
import soundfile as sf

from modelscope.fileio import File
from modelscope.metainfo import Pipelines
from modelscope.models.audio.aec.jaec import JAECModel
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks


@PIPELINES.register_module(
    Tasks.acoustic_echo_cancellation,
    module_name=Pipelines.speech_jaec_aec_16k)
class JAECPipeline(Pipeline):
    """Offline JAEC for aligned 16 kHz PCM16 mic/ref audio.

    The output retains JAEC's fixed 352-sample (22 ms) algorithmic delay.

    Pass trust_native_code=True to authorize loading the native library.
    The pipeline factory resolves the model ID and model_revision to a local
    snapshot. When specifying pipeline_name explicitly, or constructing this
    class directly, pass a local directory obtained with snapshot_download.
    """

    def __init__(self, model, trust_native_code=False, **kwargs):
        kwargs.pop('device', None)
        kwargs.pop('auto_collate', None)
        if isinstance(model, str):
            if trust_native_code is not True:
                raise RuntimeError(
                    'JAEC loads a native library from the model repository. '
                    'Pass trust_native_code=True only when you trust that '
                    'repository.')
            if not os.path.isdir(model):
                raise ValueError(
                    'JAEC requires a local model snapshot. Use pipeline() '
                    'without pipeline_name, or download the requested '
                    'revision with snapshot_download() first.')
            model = JAECModel(model, trust_native_code=True)
        elif not isinstance(model, JAECModel):
            raise TypeError('model must be a JAEC model directory')
        super().__init__(
            model=model, device='cpu', auto_collate=False, **kwargs)

    @staticmethod
    def _read_audio(source: Any, name: str) -> Tuple[np.ndarray, int]:
        if isinstance(source, bytes):
            file_bytes = source
        elif isinstance(source, str):
            file_bytes = File.read(source)
        elif isinstance(source, np.ndarray):
            if source.dtype != np.int16 or source.ndim != 1:
                raise ValueError(
                    f'{name} numpy input must be a mono int16 array')
            return np.ascontiguousarray(source), len(source)
        else:
            raise TypeError(
                f'{name} must be a WAV path, URL, bytes, or a mono int16 '
                f'array; got {type(source).__name__}')

        try:
            with sf.SoundFile(io.BytesIO(file_bytes)) as audio_file:
                if audio_file.format not in ('WAV', 'WAVEX'):
                    raise ValueError(f'{name} must be a WAV file')
                if audio_file.samplerate != 16000:
                    raise ValueError(f'{name} must use a 16 kHz sample rate')
                if audio_file.channels != 1:
                    raise ValueError(f'{name} must be mono')
                if audio_file.subtype != 'PCM_16':
                    raise ValueError(f'{name} must use PCM16 encoding')
                audio = audio_file.read(dtype='int16', always_2d=False)
        except RuntimeError as error:
            raise ValueError(f'failed to read {name} as WAV audio') from error
        return np.ascontiguousarray(audio), len(audio)

    def preprocess(self, inputs: Any, **preprocess_params) -> Dict[str, Any]:
        if isinstance(inputs, tuple):
            if len(inputs) != 2:
                raise ValueError('JAEC tuple input must contain two audios')
            inputs = {'nearend_mic': inputs[0], 'farend_speech': inputs[1]}
        elif not isinstance(inputs, dict):
            raise TypeError(
                'JAEC input must contain nearend_mic and farend_speech')
        missing = {'nearend_mic', 'farend_speech'}.difference(inputs.keys())
        if missing:
            raise ValueError(
                f'JAEC input is missing: {", ".join(sorted(missing))}')

        mic, mic_samples = self._read_audio(inputs['nearend_mic'],
                                            'nearend_mic')
        ref, ref_samples = self._read_audio(inputs['farend_speech'],
                                            'farend_speech')
        if mic_samples == 0:
            raise ValueError('JAEC input audio must not be empty')
        if mic_samples != ref_samples:
            raise ValueError(
                'nearend_mic and farend_speech must have the same length')

        padding = (-mic_samples) % 160
        if padding:
            mic = np.pad(mic, (0, padding))
            ref = np.pad(ref, (0, padding))
        return {'mic': mic, 'ref': ref, 'samples': mic_samples}

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, bytes]:
        output = self.model.process_audio(inputs['mic'], inputs['ref'])
        return {
            OutputKeys.OUTPUT_PCM:
            np.ascontiguousarray(output[:inputs['samples']]).tobytes()
        }

    def postprocess(self, inputs: Dict[str, bytes],
                    **kwargs) -> Dict[str, bytes]:
        output_path = kwargs.get('output_path')
        if output_path is not None:
            sf.write(
                output_path,
                np.frombuffer(inputs[OutputKeys.OUTPUT_PCM], dtype=np.int16),
                16000,
                subtype='PCM_16')
        return inputs
