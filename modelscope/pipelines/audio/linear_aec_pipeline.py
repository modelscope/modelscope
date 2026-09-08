# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from collections.abc import Mapping
from typing import Any, Dict

import numpy as np
import scipy.io.wavfile as wav
import torch
import yaml

from modelscope.metainfo import Pipelines
from modelscope.models.audio.aec.network.se_net import MaskNet
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LinearAECAndFbank
from modelscope.utils.architecture import (ArchitectureConfigError,
                                           instantiate_registered_architecture,
                                           require_trust_remote_code,
                                           validate_mapping_schema)
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

FEATURE_MVN = 'feature.DEY.mvn.txt'

CONFIG_YAML = 'dey_mini.yaml'

_AEC_NNET_ARCHITECTURES = {
    'mask_net': lambda: MaskNet,
}
_AEC_LEGACY_NNETS = {
    ('modelscope.models.audio.aec.network.se_net', 'MaskNet'): 'mask_net',
}
_AEC_IO_FIELDS = {
    'mask_on',
    'use_nearend_mic',
    'use_out_linear',
    'use_out_ref',
    'use_out_echo',
    'linear_aec_delay',
    'linear_aec_block',
    'feature_size',
    'mitaec_library',
    'fbank_config',
    'feat_type',
    'mvn',
}
_AEC_FBANK_FIELDS = {
    'dither',
    'frame_length',
    'frame_shift',
    'num_mel_bins',
    'sample_frequency',
    'window_type',
}


def _validate_aec_config(config: Any) -> Mapping[str, Any]:
    config = validate_mapping_schema(
        config,
        required={'io', 'nnet', 'loss'},
        optional=set(),
        context=CONFIG_YAML)
    io_config = validate_mapping_schema(
        config['io'],
        required={'fbank_config'},
        optional=_AEC_IO_FIELDS - {'fbank_config'},
        context=f'{CONFIG_YAML}.io')
    validate_mapping_schema(
        io_config['fbank_config'],
        required=set(),
        optional=_AEC_FBANK_FIELDS,
        context=f'{CONFIG_YAML}.io.fbank_config')

    nnet_config = config['nnet']
    if not isinstance(nnet_config, Mapping):
        raise ArchitectureConfigError(f'{CONFIG_YAML}.nnet must be a mapping.')
    if 'architecture' in nnet_config:
        nnet_config = validate_mapping_schema(
            nnet_config,
            required={'architecture', 'args'},
            optional=set(),
            context=f'{CONFIG_YAML}.nnet')
        architecture = nnet_config['architecture']
    else:
        nnet_config = validate_mapping_schema(
            nnet_config,
            required={'module', 'main', 'args'},
            optional=set(),
            context=f'{CONFIG_YAML}.nnet')
        architecture = _AEC_LEGACY_NNETS.get(
            (nnet_config['module'], nnet_config['main']))
        if architecture is None:
            raise ArchitectureConfigError(
                f'{CONFIG_YAML}.nnet contains an unsupported legacy architecture.'
            )
    if architecture not in _AEC_NNET_ARCHITECTURES:
        raise ArchitectureConfigError(
            f'{CONFIG_YAML}.nnet.architecture {architecture!r} is not approved.'
        )
    if not isinstance(nnet_config['args'], Mapping):
        raise ArchitectureConfigError(
            f'{CONFIG_YAML}.nnet.args must be a mapping.')

    loss_config = validate_mapping_schema(
        config['loss'],
        required={'module', 'main', 'args'},
        optional=set(),
        context=f'{CONFIG_YAML}.loss')
    if (loss_config['module'], loss_config['main']) != ('network.loss',
                                                        'mask_loss_function'):
        raise ArchitectureConfigError(
            f'{CONFIG_YAML}.loss must use the built-in mask loss metadata.')
    validate_mapping_schema(
        loss_config['args'],
        required={'loss_func', 'n_fft', 'hop_length'},
        optional=set(),
        context=f'{CONFIG_YAML}.loss.args')

    normalized = dict(config)
    normalized['nnet'] = {
        'architecture': architecture,
        'args': nnet_config['args'],
    }
    return normalized


def initialize_config(module_cfg: Mapping[str, Any]):
    """Construct an AEC network from the restricted architecture registry."""
    return instantiate_registered_architecture(
        {
            'target': module_cfg['architecture'],
            'params': module_cfg['args'],
        },
        _AEC_NNET_ARCHITECTURES,
        context=f'{CONFIG_YAML}.nnet')


@PIPELINES.register_module(
    Tasks.acoustic_echo_cancellation,
    module_name=Pipelines.speech_dfsmn_aec_psm_16k)
class LinearAECPipeline(Pipeline):
    r"""AEC Inference Pipeline only support 16000 sample rate.

    When invoke the class with pipeline.__call__(), you should provide two params:
        Dict[str, Any]
            the path of wav files, eg:{
            "nearend_mic": "/your/data/near_end_mic_audio.wav",
            "farend_speech": "/your/data/far_end_speech_audio.wav"}
        output_path (str, optional): "/your/output/audio_after_aec.wav"
            the file path to write generate audio.
    """

    def __init__(self, model, **kwargs):
        require_trust_remote_code(
            bool(kwargs.get('trust_remote_code', False)), 'LinearAECPipeline')
        super().__init__(model=model, **kwargs)

        self.use_cuda = torch.cuda.is_available()
        with open(
                os.path.join(self.model, CONFIG_YAML), encoding='utf-8') as f:
            self.config = _validate_aec_config(yaml.safe_load(f))
            self.config['io']['mvn'] = os.path.join(self.model, FEATURE_MVN)
        self._init_model()
        self.preprocessor = LinearAECAndFbank(self.config['io'])

        n_fft = self.config['loss']['args']['n_fft']
        hop_length = self.config['loss']['args']['hop_length']
        winlen = n_fft
        window = torch.hamming_window(winlen, periodic=False)

        def stft(x):
            return torch.view_as_real(
                torch.stft(
                    x,
                    n_fft,
                    hop_length,
                    winlen,
                    center=False,
                    window=window.to(x.device),
                    return_complex=True))

        def istft(x, slen):
            return torch.istft(
                torch.view_as_complex(x),
                n_fft,
                hop_length,
                winlen,
                window=window.to(x.device),
                center=False,
                length=slen)

        self.stft = stft
        self.istft = istft

    def _init_model(self):
        checkpoint = torch.load(
            os.path.join(self.model, ModelFile.TORCH_MODEL_BIN_FILE),
            map_location='cpu',
            weights_only=True)
        self.model = initialize_config(self.config['nnet'])
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model.load_state_dict(checkpoint)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        r"""The AEC process.

        Args:
            inputs: dict={'feature': Tensor, 'base': Tensor}
                'feature' feature of input audio.
                'base' the base audio to mask.

        Returns:
            output_pcm: generated audio array
        """
        output_data = self._process(inputs['feature'], inputs['base'])
        output_data = output_data.astype(np.int16).tobytes()
        return {OutputKeys.OUTPUT_PCM: output_data}

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        r"""The post process. Will save audio to file, if the output_path is given.

        Args:
            inputs: a dict contains following keys:
                - output_pcm: generated audio array
            kwargs: accept 'output_path' which is the path to write generated audio

        Returns:
            output_pcm: generated audio array
        """
        if 'output_path' in kwargs.keys():
            wav.write(
                kwargs['output_path'], self.preprocessor.SAMPLE_RATE,
                np.frombuffer(inputs[OutputKeys.OUTPUT_PCM], dtype=np.int16))
        return inputs

    def _process(self, fbanks, mixture):
        if self.use_cuda:
            fbanks = fbanks.cuda()
            mixture = mixture.cuda()
        if self.model.vad:
            with torch.no_grad():
                masks, vad = self.model(fbanks.unsqueeze(0))
                masks = masks.permute([2, 1, 0])
        else:
            with torch.no_grad():
                masks = self.model(fbanks.unsqueeze(0))
                masks = masks.permute([2, 1, 0])
        spectrum = self.stft(mixture)
        masked_spec = spectrum * masks
        masked_sig = self.istft(masked_spec, len(mixture)).cpu().numpy()
        return masked_sig
