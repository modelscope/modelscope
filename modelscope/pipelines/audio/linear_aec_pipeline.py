import importlib
import os
from typing import Any, Dict

import numpy as np
import scipy.io.wavfile as wav
import torch
import yaml

from modelscope.fileio import File
from modelscope.metainfo import Pipelines
from modelscope.preprocessors.audio import LinearAECAndFbank
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from ..base import Pipeline
from ..builder import PIPELINES
from ..outputs import OutputKeys

logger = get_logger()

FEATURE_MVN = 'feature.DEY.mvn.txt'

CONFIG_YAML = 'dey_mini.yaml'

AEC_LIB_URL = 'https://modelscope.oss-cn-beijing.aliyuncs.com/dependencies/ics_MaaS_AEC_lib_libmitaec_pyio.so'
AEC_LIB_FILE = 'libmitaec_pyio.so'


def initialize_config(module_cfg):
    r"""According to config items, load specific module dynamically with params.
        1. Load the module corresponding to the "module" param.
        2. Call function (or instantiate class) corresponding to the "main" param.
        3. Send the param (in "args") into the function (or class) when calling ( or instantiating).

    Args:
        module_cfg (dict): config items, eg:
            {
                "module": "models.model",
                "main": "Model",
                "args": {...}
            }

    Returns:
        the module loaded.
    """
    module = importlib.import_module(module_cfg['module'])
    return getattr(module, module_cfg['main'])(**module_cfg['args'])


@PIPELINES.register_module(
    Tasks.speech_signal_process,
    module_name=Pipelines.speech_dfsmn_aec_psm_16k)
class LinearAECPipeline(Pipeline):
    r"""AEC Inference Pipeline only support 16000 sample rate.

    When invoke the class with pipeline.__call__(), you should provide two params:
        Dict[str, Any]
            the path of wav files，eg:{
            "nearend_mic": "/your/data/near_end_mic_audio.wav",
            "farend_speech": "/your/data/far_end_speech_audio.wav"}
        output_path (str, optional): "/your/output/audio_after_aec.wav"
            the file path to write generate audio.
    """

    def __init__(self, model):
        r"""
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model)

        # auto download so for linux inference before light-weight docker got ready
        if not os.path.exists(AEC_LIB_FILE):
            logger.info(f'downloading {AEC_LIB_URL} to {AEC_LIB_FILE}')
            with open(AEC_LIB_FILE, 'wb') as ofile:
                ofile.write(File.read(AEC_LIB_URL))

        self.use_cuda = torch.cuda.is_available()
        with open(
                os.path.join(self.model, CONFIG_YAML), encoding='utf-8') as f:
            self.config = yaml.full_load(f.read())
            self.config['io']['mvn'] = os.path.join(self.model, FEATURE_MVN)
        self._init_model()
        self.preprocessor = LinearAECAndFbank(self.config['io'])

        n_fft = self.config['loss']['args']['n_fft']
        hop_length = self.config['loss']['args']['hop_length']
        winlen = n_fft
        window = torch.hamming_window(winlen, periodic=False)

        def stft(x):
            return torch.stft(
                x,
                n_fft,
                hop_length,
                winlen,
                center=False,
                window=window.to(x.device),
                return_complex=False)

        def istft(x, slen):
            return torch.istft(
                x,
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
            map_location='cpu')
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
            dict:
                {
                    'output_pcm': generated audio array
                }
        """
        output_data = self._process(inputs['feature'], inputs['base'])
        return {OutputKeys.OUTPUT_PCM: output_data}

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        r"""The post process. Will save audio to file, if the output_path is given.

        Args:
            inputs: dict:
                {
                    'output_pcm': generated audio array
                }
            kwargs: accept 'output_path' which is the path to write generated audio

        Returns:
            dict:
                {
                    'output_pcm': generated audio array
                }
        """
        if 'output_path' in kwargs.keys():
            wav.write(kwargs['output_path'], self.preprocessor.SAMPLE_RATE,
                      inputs[OutputKeys.OUTPUT_PCM].astype(np.int16))
        inputs[OutputKeys.OUTPUT_PCM] = inputs[OutputKeys.OUTPUT_PCM] / 32768.0
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
