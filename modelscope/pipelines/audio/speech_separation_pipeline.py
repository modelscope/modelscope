# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, List, Sequence, Tuple, Union

import json
import yaml
from funasr.utils import asr_utils

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.audio.audio_utils import (generate_scp_from_url,
                                                update_local_model)
from modelscope.utils.constant import Frameworks, ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['SeparationPipeline']


@PIPELINES.register_module(
    Tasks.speech_separation, module_name=Pipelines.funasr_speech_separation)
class SeparationPipeline(Pipeline):
    """Speech Separation Inference Pipeline
    use `model` to create a speech separation pipeline for prediction.

    Args:
        model: A model instance, or a model local dir, or a model id in the model hub.
        kwargs (dict, `optional`):
            Extra kwargs passed into the preprocessor's constructor.

    Example:
        >>> from modelscope.pipelines import pipeline
        >>> pipeline = pipeline(
        >>>    task=Tasks.speech_separation, model='damo/speech_separation_mossformer_8k_pytorch')
        >>> audio_in = 'mix_speech.wav'
        >>> print(pipeline(audio_in))

    """

    def __init__(self,
                 model: Union[Model, str] = None,
                 ngpu: int = 1,
                 **kwargs):
        """use `model` to create an speech separation pipeline for prediction
        """
        super().__init__(model=model, **kwargs)
        config_path = os.path.join(model, ModelFile.CONFIGURATION)
        self.cmd = self.get_cmd(config_path, kwargs, model)

        from funasr.bin import ss_inference_launch
        self.funasr_infer_modelscope = ss_inference_launch.inference_launch(
            mode=self.cmd['mode'],
            batch_size=self.cmd['batch_size'],
            ngpu=ngpu,
            log_level=self.cmd['log_level'],
            ss_infer_config=self.cmd['ss_infer_config'],
            ss_model_file=self.cmd['ss_model_file'],
            output_dir=self.cmd['output_dir'],
            dtype=self.cmd['dtype'],
            seed=self.cmd['seed'],
            num_workers=self.cmd['num_workers'],
            num_spks=self.cmd['num_spks'],
            param_dict=self.cmd['param_dict'],
            **kwargs,
        )

    def __call__(self,
                 audio_in: Union[str, bytes],
                 audio_fs: int = None,
                 recog_type: str = None,
                 audio_format: str = None,
                 output_dir: str = None,
                 param_dict: dict = None,
                 **kwargs) -> Dict[str, Any]:
        """
        Decoding the input audios
        Args:
            audio_in('str' or 'bytes'):
                - A string containing a local path to a wav file
                - A string containing a local path to a scp
                - A string containing a wav url
                - A bytes input
            audio_fs('int'):
                frequency of sample
            recog_type('str'):
                recog type for wav file or datasets file ('wav', 'test', 'dev', 'train')
            audio_format('str'):
                audio format ('pcm', 'scp', 'kaldi_ark', 'tfrecord')
            output_dir('str'):
                output dir
            param_dict('dict'):
                extra kwargs
        Return:
            A dictionary of result or a list of dictionary of result.

            The dictionary contain the following keys:
            - **text** ('str') --The vad result.
        """
        self.audio_in = None
        self.raw_inputs = None
        self.recog_type = recog_type
        self.audio_format = audio_format
        self.audio_fs = None
        checking_audio_fs = None
        if output_dir is not None:
            self.cmd['output_dir'] = output_dir
        if param_dict is not None:
            self.cmd['param_dict'] = param_dict
        if isinstance(audio_in, str):
            # for funasr code, generate wav.scp from url or local path
            self.audio_in, self.raw_inputs = generate_scp_from_url(audio_in)
        elif isinstance(audio_in, bytes):
            self.audio_in = audio_in
            self.raw_inputs = None
        else:
            import numpy
            import torch
            if isinstance(audio_in, torch.Tensor):
                self.audio_in = None
                self.raw_inputs = audio_in
            elif isinstance(audio_in, numpy.ndarray):
                self.audio_in = None
                self.raw_inputs = audio_in

        # set the sample_rate of audio_in if checking_audio_fs is valid
        if checking_audio_fs is not None:
            self.audio_fs = checking_audio_fs

        if recog_type is None or audio_format is None:
            self.recog_type, self.audio_format, self.audio_in = asr_utils.type_checking(
                audio_in=self.audio_in,
                recog_type=recog_type,
                audio_format=audio_format)

        if hasattr(asr_utils,
                   'sample_rate_checking') and self.audio_in is not None:
            checking_audio_fs = asr_utils.sample_rate_checking(
                self.audio_in, self.audio_format)
            if checking_audio_fs is not None:
                self.audio_fs = checking_audio_fs
        if audio_fs is not None:
            self.cmd['fs']['audio_fs'] = audio_fs
        else:
            self.cmd['fs']['audio_fs'] = self.audio_fs

        output = self.forward(self.audio_in, **kwargs)
        return output

    def get_cmd(self, config_path, extra_args, model_path) -> Dict[str, Any]:
        model_cfg = json.loads(open(config_path).read())
        model_dir = os.path.dirname(config_path)
        # generate inference command
        ss_model_path = os.path.join(
            model_dir, model_cfg['model']['model_config']['ss_model_name'])
        ss_model_config = os.path.join(
            model_dir, model_cfg['model']['model_config']['ss_model_config'])
        mode = model_cfg['model']['model_config']['mode']
        frontend_conf = None
        if os.path.exists(ss_model_config):
            config_file = open(ss_model_config, encoding='utf-8')
            root = yaml.full_load(config_file)
            config_file.close()
            if 'frontend_conf' in root:
                frontend_conf = root['frontend_conf']
        update_local_model(model_cfg['model']['model_config'], model_path,
                           extra_args)

        cmd = {
            'mode': mode,
            'batch_size': 1,
            'ngpu': 1,  # 0: only CPU, ngpu>=1: gpu number if cuda is available
            'log_level': 'ERROR',
            'ss_infer_config': ss_model_config,
            'ss_model_file': ss_model_path,
            'output_dir': None,
            'dtype': 'float32',
            'seed': 0,
            'num_workers': 0,
            'num_spks': 2,
            'param_dict': None,
            'fs': {
                'model_fs': None,
                'audio_fs': None
            }
        }
        if frontend_conf is not None and 'fs' in frontend_conf:
            cmd['fs']['model_fs'] = frontend_conf['fs']

        user_args_dict = [
            'output_dir', 'batch_size', 'mode', 'ngpu', 'param_dict',
            'num_workers', 'fs'
        ]

        for user_args in user_args_dict:
            if user_args in extra_args:
                if extra_args.get(user_args) is not None:
                    cmd[user_args] = extra_args[user_args]
                del extra_args[user_args]

        return cmd

    def postprocess(self, inputs: Dict[str, Any],
                    **post_params) -> Dict[str, Any]:
        return inputs

    def forward(self, audio_in: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Decoding
        """
        logger.info('Speech Separation Processing ...')
        # generate inputs
        data_cmd: Sequence[Tuple[str, str, str]]
        if isinstance(self.audio_in, bytes):
            data_cmd = [self.audio_in, 'speech', 'bytes']
        elif isinstance(self.audio_in, str):
            data_cmd = [self.audio_in, 'speech', 'sound']
        elif self.raw_inputs is not None:
            data_cmd = None
        self.cmd['name_and_type'] = data_cmd
        self.cmd['raw_inputs'] = self.raw_inputs
        self.cmd['audio_in'] = self.audio_in

        ss_result = self.run_inference(self.cmd, **kwargs)

        return ss_result

    def run_inference(self, cmd, **kwargs):
        ss_result = []
        if self.framework == Frameworks.torch:
            ss_result = self.funasr_infer_modelscope(
                data_path_and_name_and_type=cmd['name_and_type'],
                raw_inputs=cmd['raw_inputs'],
                output_dir_v2=cmd['output_dir'],
                fs=cmd['fs'],
                param_dict=cmd['param_dict'],
                **kwargs)
        else:
            raise ValueError('model type is mismatching')

        return ss_result
