# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import yaml

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.audio.audio_utils import (generate_scp_from_url,
                                                update_local_model)
from modelscope.utils.constant import Frameworks, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['AudioQuantizationPipeline']


@PIPELINES.register_module(
    Tasks.audio_quantization,
    module_name=Pipelines.audio_quantization_inference)
class AudioQuantizationPipeline(Pipeline):
    """Audio Quantization Inference Pipeline
    use `model` to create a audio quantization pipeline.

    Args:
        model (AudioQuantizationPipeline): A model instance, or a model local dir, or a model id in the model hub.
        kwargs (dict, `optional`):
            Extra kwargs passed into the preprocessor's constructor.
    Examples:
        >>> from modelscope.pipelines import pipeline
        >>> from modelscope.utils.constant import Tasks
        >>> pipeline_aq = pipeline(
        >>>    task=Tasks.audio_quantization,
        >>>    model='damo/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch'
        >>> )
        >>> audio_in='example.wav'
        >>> print(pipeline_aq(audio_in))

    """

    def __init__(self,
                 model: Union[Model, str] = None,
                 ngpu: int = 1,
                 **kwargs):
        """use `model` to create an asr pipeline for prediction
        """
        super().__init__(model=model, **kwargs)
        self.model_cfg = self.model.forward()
        self.cmd = self.get_cmd(kwargs, model)

        from funcodec.bin import codec_inference
        self.funasr_infer_modelscope = codec_inference.inference_modelscope(
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
            sampling_rate=self.cmd['sampling_rate'],
            bit_width=self.cmd['bit_width'],
            use_scale=self.cmd['use_scale'],
            param_dict=self.cmd['param_dict'],
            **kwargs,
        )

    def __call__(self,
                 audio_in: Union[tuple, str, Any] = None,
                 output_dir: str = None,
                 param_dict: dict = None) -> Dict[str, Any]:
        if len(audio_in) == 0:
            raise ValueError('The input should not be null.')
        else:
            self.audio_in = audio_in
        if output_dir is not None:
            self.cmd['output_dir'] = output_dir
        self.cmd['param_dict'] = param_dict

        output = self.forward(self.audio_in)
        result = self.postprocess(output)
        return result

    def postprocess(self, inputs: list) -> Dict[str, Any]:
        """Postprocessing
        """
        rst = {}
        for i in range(len(inputs)):
            if len(inputs) == 1 and i == 0:
                recon_wav = inputs[0]['value']
                output_wav = recon_wav.cpu().numpy()[0]
                output_wav = (output_wav * (2**15)).astype(np.int16)
                rst[OutputKeys.OUTPUT_WAV] = output_wav
            else:
                # for multiple inputs
                rst[inputs[i]['key']] = inputs[i]['value']
        return rst

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
            'sampling_rate': 16000,
            'bit_width': 8000,
            'use_scale': True,
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

    def forward(self, audio_in: Union[tuple, str, Any] = None) -> list:
        """Decoding
        """
        # log  file_path/url or tuple (str, str)
        if isinstance(audio_in, str):
            logger.info(f'Audio Quantization Processing: {audio_in} ...')
        else:
            logger.info(
                f'Audio Quantization Processing: {str(audio_in)[:100]} ...')

        data_cmd, raw_inputs = None, None
        if isinstance(audio_in, str):
            # for scp inputs
            if len(audio_in.split(',')) == 3:
                data_cmd = [tuple(audio_in.split(','))]
            # for single-file inputs
            else:
                audio_scp, _ = generate_scp_from_url(audio_in)
                raw_inputs = audio_scp
        # for raw bytes
        elif isinstance(audio_in, bytes):
            data_cmd = (audio_in, 'speech', 'bytes')
        # for ndarray and tensor inputs
        else:
            import torch
            import numpy as np
            if isinstance(audio_in, torch.Tensor):
                raw_inputs = audio_in
            elif isinstance(audio_in, np.ndarray):
                raw_inputs = audio_in
            else:
                raise TypeError('Unsupported data type.')

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
