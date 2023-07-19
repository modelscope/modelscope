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

__all__ = ['TimestampPipeline']


@PIPELINES.register_module(
    Tasks.speech_timestamp, module_name=Pipelines.speech_timestamp_inference)
class TimestampPipeline(Pipeline):
    """Timestamp Inference Pipeline
    Example:

    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.utils.constant import Tasks

    >>> pipeline_infer = pipeline(
    >>>    task=Tasks.speech_timestamp,
    >>>    model='damo/speech_timestamp_predictor-v1-16k-offline')

    >>> audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_timestamps.wav'
    >>> text_in='一 个 东 太 平 洋 国 家 为 什 么 跑 到 西 太 平 洋 来 了 呢'
    >>> print(pipeline_infer(audio_in, text_in))

    """

    def __init__(self,
                 model: Union[Model, str] = None,
                 ngpu: int = 1,
                 **kwargs):
        """
        Use `model` and `preprocessor` to create an asr pipeline for prediction
        Args:
            model ('Model' or 'str'):
                The pipeline handles three types of model:

                - A model instance
                - A model local dir
                - A model id in the model hub
            output_dir('str'):
                output dir path
            batch_size('int'):
                the batch size for inference
            ngpu('int'):
                the number of gpus, 0 indicates CPU mode
            split_with_space('bool'):
                split the input sentence by space
            seg_dict_file('str'):
                seg dict file
            param_dict('dict'):
                extra kwargs
        """
        super().__init__(model=model, **kwargs)
        config_path = os.path.join(model, ModelFile.CONFIGURATION)
        self.cmd = self.get_cmd(config_path, kwargs, model)

        from funasr.bin import tp_inference_launch
        self.funasr_infer_modelscope = tp_inference_launch.inference_launch(
            mode=self.cmd['mode'],
            batch_size=self.cmd['batch_size'],
            dtype=self.cmd['dtype'],
            ngpu=ngpu,
            seed=self.cmd['seed'],
            num_workers=self.cmd['num_workers'],
            log_level=self.cmd['log_level'],
            key_file=self.cmd['key_file'],
            timestamp_infer_config=self.cmd['timestamp_infer_config'],
            timestamp_model_file=self.cmd['timestamp_model_file'],
            timestamp_cmvn_file=self.cmd['timestamp_cmvn_file'],
            output_dir=self.cmd['output_dir'],
            allow_variable_data_keys=self.cmd['allow_variable_data_keys'],
            split_with_space=self.cmd['split_with_space'],
            seg_dict_file=self.cmd['seg_dict_file'],
            param_dict=self.cmd['param_dict'],
            **kwargs,
        )

    def __call__(self,
                 audio_in: Union[str, bytes],
                 text_in: str,
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
            text_in('str'):
                - A text str input
                - A local text file input endswith .txt or .scp
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
            - **text** ('str') --The timestamp result.
        """
        self.audio_in = None
        self.text_in = None
        self.raw_inputs = None
        self.recog_type = recog_type
        self.audio_format = audio_format
        self.audio_fs = None
        checking_audio_fs = None
        if output_dir is not None:
            self.cmd['output_dir'] = output_dir
        if param_dict is not None:
            self.cmd['param_dict'] = param_dict

        # audio
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
        # text
        if text_in.startswith('http'):
            self.text_in, _ = generate_text_from_url(text_in)
        else:
            self.text_in = text_in

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

        output = self.forward(self.audio_in, self.text_in, **kwargs)
        result = self.postprocess(output)
        return result

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocessing
        """
        rst = {}
        for i in range(len(inputs)):
            if i == 0:
                for key, value in inputs[0].items():
                    if key == 'value':
                        if len(value) > 0:
                            rst[OutputKeys.TEXT] = value
                    elif key != 'key':
                        rst[key] = value
            else:
                rst[inputs[i]['key']] = inputs[i]['value']
        return rst

    def get_cmd(self, config_path, extra_args, model_path) -> Dict[str, Any]:
        model_cfg = json.loads(open(config_path).read())
        model_dir = os.path.dirname(config_path)
        # generate inference command
        timestamp_model_file = os.path.join(
            model_dir,
            model_cfg['model']['model_config']['timestamp_model_file'])
        timestamp_infer_config = os.path.join(
            model_dir,
            model_cfg['model']['model_config']['timestamp_infer_config'])
        timestamp_cmvn_file = os.path.join(
            model_dir,
            model_cfg['model']['model_config']['timestamp_cmvn_file'])
        mode = model_cfg['model']['model_config']['mode']
        frontend_conf = None
        if os.path.exists(timestamp_infer_config):
            config_file = open(timestamp_infer_config, encoding='utf-8')
            root = yaml.full_load(config_file)
            config_file.close()
            if 'frontend_conf' in root:
                frontend_conf = root['frontend_conf']
        seg_dict_file = None
        if 'seg_dict_file' in model_cfg['model']['model_config']:
            seg_dict_file = os.path.join(
                model_dir, model_cfg['model']['model_config']['seg_dict_file'])
        update_local_model(model_cfg['model']['model_config'], model_path,
                           extra_args)

        cmd = {
            'mode': mode,
            'batch_size': 1,
            'dtype': 'float32',
            'ngpu': 0,  # 0: only CPU, ngpu>=1: gpu number if cuda is available
            'seed': 0,
            'num_workers': 0,
            'log_level': 'ERROR',
            'key_file': None,
            'allow_variable_data_keys': False,
            'split_with_space': True,
            'seg_dict_file': seg_dict_file,
            'timestamp_infer_config': timestamp_infer_config,
            'timestamp_model_file': timestamp_model_file,
            'timestamp_cmvn_file': timestamp_cmvn_file,
            'output_dir': None,
            'param_dict': None,
            'fs': {
                'model_fs': None,
                'audio_fs': None
            }
        }
        if frontend_conf is not None and 'fs' in frontend_conf:
            cmd['fs']['model_fs'] = frontend_conf['fs']

        user_args_dict = [
            'output_dir',
            'batch_size',
            'mode',
            'ngpu',
            'param_dict',
            'num_workers',
            'log_level',
            'split_with_space',
            'seg_dict_file',
        ]

        for user_args in user_args_dict:
            if user_args in extra_args:
                if extra_args.get(user_args) is not None:
                    cmd[user_args] = extra_args[user_args]
                del extra_args[user_args]

        return cmd

    def forward(self, audio_in: Dict[str, Any], text_in: Dict[str, Any],
                **kwargs) -> Dict[str, Any]:
        """Decoding
        """
        logger.info('Timestamp Processing ...')
        # generate inputs
        data_cmd: Sequence[Tuple[str, str, str]]
        if isinstance(self.audio_in, bytes):
            data_cmd = [(self.audio_in, 'speech', 'bytes')]
            data_cmd.append((text_in, 'text', 'text'))
        elif isinstance(self.audio_in, str):
            data_cmd = [(self.audio_in, 'speech', 'sound')]
            data_cmd.append((text_in, 'text', 'text'))
        elif self.raw_inputs is not None:
            data_cmd = None

        if self.raw_inputs is None and data_cmd is None:
            raise ValueError('please check audio_in')

        self.cmd['name_and_type'] = data_cmd
        self.cmd['raw_inputs'] = self.raw_inputs
        self.cmd['audio_in'] = self.audio_in

        tp_result = self.run_inference(self.cmd, **kwargs)

        return tp_result

    def run_inference(self, cmd, **kwargs):
        tp_result = []
        if self.framework == Frameworks.torch:
            tp_result = self.funasr_infer_modelscope(
                data_path_and_name_and_type=cmd['name_and_type'],
                raw_inputs=cmd['raw_inputs'],
                output_dir_v2=cmd['output_dir'],
                fs=cmd['fs'],
                param_dict=cmd['param_dict'],
                **kwargs)
        else:
            raise ValueError('model type is mismatching')

        return tp_result
