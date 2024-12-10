# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Any, Dict, Optional, Union

import json
import numpy as np

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.audio.audio_utils import (generate_scp_from_url,
                                                update_local_model)
from modelscope.utils.constant import Frameworks, ModelFile, Tasks
from modelscope.utils.hub import snapshot_download
from modelscope.utils.logger import get_logger

__all__ = ['LauraCodecTTSPipeline']

logger = get_logger()


@PIPELINES.register_module(
    Tasks.text_to_speech, module_name=Pipelines.laura_codec_tts_inference)
class LauraCodecTTSPipeline(Pipeline):
    """Laura-style Codec-based TTS Inference Pipeline
    use `model` to create a TTS pipeline.

    Args:
        model (LauraCodecTTSPipeline): A model instance, or a model local dir, or a model id in the model hub.
        kwargs (dict, `optional`):
            Extra kwargs passed into the preprocessor's constructor.
    Examples:
        >>> from modelscope.pipelines import pipeline
        >>> from modelscope.utils.constant import Tasks
        >>> my_pipeline = pipeline(
        >>>    task=Tasks.text_to_speech,
        >>>    model='damo/speech_synthesizer-laura-en-libritts-16k-codec_nq2-pytorch'
        >>> )
        >>> text='nothing was to be done but to put about, and return in disappointment towards the north.'
        >>> prompt_text='one of these is context'
        >>> prompt_speech='example/prompt.wav'
        >>> print(my_pipeline(text))

    """

    def __init__(self,
                 model: Union[Model, str] = None,
                 codec_model: Optional[Union[Model, str]] = None,
                 codec_model_revision: Optional[str] = None,
                 ngpu: int = 1,
                 **kwargs):
        """use `model` to create an asr pipeline for prediction
        """
        super().__init__(model=model, **kwargs)
        self.model_cfg = self.model.forward()
        self.codec_model = codec_model
        self.codec_model_revision = codec_model_revision
        self.cmd = self.get_cmd(kwargs, model)

        from funcodec.bin import text2audio_inference
        self.funasr_infer_modelscope = text2audio_inference.inference_func(
            mode=self.cmd['mode'],
            output_dir=self.cmd['output_dir'],
            batch_size=self.cmd['batch_size'],
            dtype=self.cmd['dtype'],
            ngpu=ngpu,
            seed=self.cmd['seed'],
            num_workers=self.cmd['num_workers'],
            log_level=self.cmd['log_level'],
            key_file=self.cmd['key_file'],
            config_file=self.cmd['config_file'],
            model_file=self.cmd['model_file'],
            model_tag=self.cmd['model_tag'],
            allow_variable_data_keys=self.cmd['allow_variable_data_keys'],
            streaming=self.cmd['streaming'],
            text_emb_model=self.cmd['text_emb_model'],
            beam_size=self.cmd['beam_size'],
            sampling=self.cmd['sampling'],
            continual=self.cmd['continual'],
            tokenize_to_phone=self.cmd['tokenize_to_phone'],
            exclude_prompt=self.cmd['exclude_prompt'],
            codec_config_file=self.cmd['codec_config_file'],
            codec_model_file=self.cmd['codec_model_file'],
            param_dict=self.cmd['param_dict'])

    def __call__(self,
                 text: Union[tuple, str, Any] = None,
                 prompt_text: Union[tuple, str, Any] = None,
                 prompt_audio: Union[tuple, str, Any] = None,
                 output_dir: str = None,
                 param_dict: dict = None) -> Dict[str, Any]:
        if len(text) == 0:
            raise ValueError('The input should not be null.')
        if output_dir is not None:
            self.cmd['output_dir'] = output_dir
        self.cmd['param_dict'] = param_dict

        output = self.forward(text, prompt_text, prompt_audio)
        result = self.postprocess(output)
        return result

    def postprocess(self, inputs: list) -> Dict[str, Any]:
        """Postprocessing
        """
        rst = {}
        for i in range(len(inputs)):
            if len(inputs) == 1 and i == 0:
                recon_wav = inputs[0]['value']['gen']
                rst[OutputKeys.OUTPUT_WAV] = recon_wav.cpu().numpy()[0]
            else:
                # for multiple inputs
                rst[inputs[i]['key']] = inputs[i]['value']['gen']
        return rst

    def load_codec_model(self, cmd):
        if self.codec_model is not None and self.codec_model != '':
            if os.path.exists(self.codec_model):
                codec_model = self.codec_model
            else:
                codec_model = snapshot_download(
                    self.codec_model, revision=self.codec_model_revision)
            logger.info('loading codec model from {0} ...'.format(codec_model))
            config_path = os.path.join(codec_model, ModelFile.CONFIGURATION)
            model_cfg = json.loads(open(config_path).read())
            model_dir = os.path.dirname(config_path)
            cmd['codec_model_file'] = os.path.join(
                model_dir, model_cfg['model']['model_config']['model_file'])
            cmd['codec_config_file'] = os.path.join(
                model_dir, model_cfg['model']['model_config']['config_file'])

    def get_cmd(self, extra_args, model_path) -> Dict[str, Any]:
        # generate asr inference command
        mode = self.model_cfg['model_config']['mode']
        _model_path = os.path.join(
            self.model_cfg['model_workspace'],
            self.model_cfg['model_config']['model_file'])
        _model_config = os.path.join(
            self.model_cfg['model_workspace'],
            self.model_cfg['model_config']['config_file'])
        update_local_model(self.model_cfg['model_config'], model_path,
                           extra_args)

        cmd = {
            'mode': mode,
            'output_dir': None,
            'batch_size': 1,
            'dtype': 'float32',
            'ngpu': 1,  # 0: only CPU, ngpu>=1: gpu number if cuda is available
            'seed': 0,
            'num_workers': 0,
            'log_level': 'ERROR',
            'key_file': None,
            'model_file': _model_path,
            'config_file': _model_config,
            'model_tag': None,
            'allow_variable_data_keys': True,
            'streaming': False,
            'beam_size': 1,
            'sampling': 25,
            'text_emb_model': None,
            'continual': True,
            'tokenize_to_phone': True,
            'exclude_prompt': True,
            'codec_model_file': None,
            'codec_config_file': None,
            'param_dict': None,
        }
        user_args_dict = [
            'output_dir',
            'batch_size',
            'ngpu',
            'log_level',
            'allow_variable_data_keys',
            'streaming',
            'num_workers',
            'sampling_rate',
            'bit_width',
            'use_scale',
            'param_dict',
        ]

        model_config = self.model_cfg['model_config']
        if model_config.__contains__(
                'codec_model') and self.codec_model is None:
            self.codec_model = model_config['codec_model']
        if model_config.__contains__(
                'codec_model_revision') and self.codec_model_revision is None:
            self.codec_model_revision = model_config['codec_model_revision']
        self.load_codec_model(cmd)

        # re-write the config with configure.json
        for user_args in user_args_dict:
            if (user_args in self.model_cfg['model_config']
                    and self.model_cfg['model_config'][user_args] is not None):
                if isinstance(cmd[user_args], dict) and isinstance(
                        self.model_cfg['model_config'][user_args], dict):
                    cmd[user_args].update(
                        self.model_cfg['model_config'][user_args])
                else:
                    cmd[user_args] = self.model_cfg['model_config'][user_args]

        # rewrite the config with user args
        for user_args in user_args_dict:
            if user_args in extra_args:
                if extra_args.get(user_args) is not None:
                    if isinstance(cmd[user_args], dict) and isinstance(
                            extra_args[user_args], dict):
                        cmd[user_args].update(extra_args[user_args])
                    else:
                        cmd[user_args] = extra_args[user_args]
                del extra_args[user_args]

        return cmd

    def forward(self,
                text: Union[tuple, str, Any] = None,
                prompt_text: Union[tuple, str, Any] = None,
                prompt_audio: Union[tuple, str, Any] = None,
                **forward_params) -> list:
        """Decoding
        """
        if isinstance(text, str):
            logger.info(f'Generate speech for: {text} ...')

        data_cmd, raw_inputs = None, None
        # process text input
        # for scp inputs
        if len(text.split(',')) == 3:
            data_cmd = [tuple(text.split(','))]
        # for single-file inputs
        else:
            raw_inputs = [text]

        if prompt_text is not None and prompt_audio is not None:
            if len(prompt_text.split(',')) == 3:
                data_cmd.append(tuple(prompt_text.split(',')))
            else:
                raw_inputs.append(prompt_text)

            if isinstance(prompt_audio, str):
                if len(prompt_audio.split(',')) == 3:
                    data_cmd.append(tuple(prompt_audio.split(',')))
                else:
                    audio_path, _ = generate_scp_from_url(prompt_audio)
                    raw_inputs.append(audio_path)
            # for ndarray and tensor inputs
            else:
                import torch
                if isinstance(prompt_audio, torch.Tensor):
                    raw_inputs.append(prompt_audio.numpy())
                elif isinstance(prompt_audio, np.ndarray):
                    raw_inputs.append(prompt_audio)
                else:
                    raise TypeError(
                        f'Unsupported prompt audio type {type(prompt_audio)}.')

        self.cmd['name_and_type'] = data_cmd
        self.cmd['raw_inputs'] = raw_inputs
        result = self.run_inference(self.cmd)

        return result

    def run_inference(self, cmd):
        if self.framework == Frameworks.torch:
            sv_result = self.funasr_infer_modelscope(
                data_path_and_name_and_type=cmd['name_and_type'],
                raw_inputs=cmd['raw_inputs'],
                output_dir_v2=cmd['output_dir'],
                param_dict=cmd['param_dict'])
        else:
            raise ValueError('model type is mismatching')

        return sv_result
