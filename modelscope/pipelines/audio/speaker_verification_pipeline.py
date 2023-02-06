# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
from typing import Any, Dict, List, Sequence, Tuple, Union

import yaml

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.audio.audio_utils import (generate_scp_for_sv,
                                                generate_sv_scp_from_url)
from modelscope.utils.constant import Frameworks, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['SpeakerVerificationPipeline']


@PIPELINES.register_module(
    Tasks.speaker_verification, module_name=Pipelines.sv_inference)
class SpeakerVerificationPipeline(Pipeline):
    """Speaker Verification Inference Pipeline
    use `model` to create a Speaker Verification pipeline.

    Args:
        model (SpeakerVerificationPipeline): A model instance, or a model local dir, or a model id in the model hub.
        kwargs (dict, `optional`):
            Extra kwargs passed into the preprocessor's constructor.
    Example:
    >>> from modelscope.pipelines import pipeline
    >>> pipeline_sv = pipeline(
    >>>    task=Tasks.speaker_verification, model='damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch')
    >>> audio_in=('','')
    >>> print(pipeline_sv(audio_in))

    """

    def __init__(self, model: Union[Model, str] = None, **kwargs):
        """use `model` to create an asr pipeline for prediction
        """
        super().__init__(model=model, **kwargs)
        self.model_cfg = self.model.forward()
        self.cmd = self.get_cmd(kwargs)

        from funasr.bin import sv_inference_launch
        self.funasr_infer_modelscope = sv_inference_launch.inference_launch(
            mode=self.cmd['mode'],
            output_dir=self.cmd['output_dir'],
            batch_size=self.cmd['batch_size'],
            dtype=self.cmd['dtype'],
            ngpu=self.cmd['ngpu'],
            seed=self.cmd['seed'],
            num_workers=self.cmd['num_workers'],
            log_level=self.cmd['log_level'],
            key_file=self.cmd['key_file'],
            sv_train_config=self.cmd['sv_train_config'],
            sv_model_file=self.cmd['sv_model_file'],
            model_tag=self.cmd['model_tag'],
            allow_variable_data_keys=self.cmd['allow_variable_data_keys'],
            streaming=self.cmd['streaming'],
            embedding_node=self.cmd['embedding_node'],
            sv_threshold=self.cmd['sv_threshold'],
            param_dict=self.cmd['param_dict'],
        )

    def __call__(self,
                 audio_in: Union[tuple, str, Any] = None,
                 output_dir: str = None,
                 param_dict: dict = None) -> Dict[str, Any]:
        if len(audio_in) == 0:
            raise ValueError('The input of sv should not be null.')
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
            if i == 0:
                if isinstance(self.audio_in, tuple) or isinstance(
                        self.audio_in, list):
                    score = inputs[0]['value']
                    rst[OutputKeys.SCORES] = score
                else:
                    embedding = inputs[0]['value']
                    rst[OutputKeys.SPK_EMBEDDING] = embedding
            rst[inputs[i]['key']] = inputs[i]['value']
        return rst

    def get_cmd(self, extra_args) -> Dict[str, Any]:
        # generate asr inference command
        mode = self.model_cfg['model_config']['mode']
        sv_model_path = self.model_cfg['sv_model_path']
        sv_model_config = os.path.join(
            self.model_cfg['model_workspace'],
            self.model_cfg['model_config']['sv_model_config'])
        cmd = {
            'mode': mode,
            'output_dir': None,
            'batch_size': 1,
            'dtype': 'float32',
            'ngpu': 1,  # 0: only CPU, ngpu>=1: gpu number if cuda is available
            'seed': 0,
            'num_workers': 1,
            'log_level': 'ERROR',
            'key_file': None,
            'sv_model_file': sv_model_path,
            'sv_train_config': sv_model_config,
            'model_tag': None,
            'allow_variable_data_keys': True,
            'streaming': False,
            'embedding_node': 'resnet1_dense',
            'sv_threshold': 0.9465,
            'param_dict': None,
        }
        user_args_dict = [
            'output_dir',
            'batch_size',
            'ngpu',
            'embedding_node',
            'sv_threshold',
            'log_level',
            'allow_variable_data_keys',
            'streaming',
            'param_dict',
        ]

        for user_args in user_args_dict:
            if user_args in extra_args and extra_args[user_args] is not None:
                cmd[user_args] = extra_args[user_args]

        return cmd

    def forward(self, audio_in: Union[tuple, str, Any] = None) -> list:
        """Decoding
        """
        logger.info(
            'Speaker Verification Processing: {0} ...'.format(audio_in))

        data_cmd, raw_inputs = None, None
        if isinstance(audio_in, tuple) or isinstance(audio_in, list):
            # generate audio_scp
            assert len(audio_in) == 2
            if isinstance(audio_in[0], str):
                # for scp inputs
                if len(audio_in[0].split(',')) == 3 and audio_in[0].split(
                        ',')[0].endswith('.scp'):
                    if len(audio_in[1].split(',')) == 3 and audio_in[1].split(
                            ',')[0].endswith('.scp'):
                        data_cmd = [
                            tuple(audio_in[0].split(',')),
                            tuple(audio_in[1].split(','))
                        ]
                # for single-file inputs
                else:
                    audio_scp_1, audio_scp_2 = generate_sv_scp_from_url(
                        audio_in)
                    data_cmd = [(audio_scp_1, 'speech', 'sound'),
                                (audio_scp_2, 'ref_speech', 'sound')]
            # for raw bytes inputs
            elif isinstance(audio_in[0], bytes):
                data_cmd = [(audio_in[0], 'speech', 'bytes'),
                            (audio_in[1], 'ref_speech', 'bytes')]
            else:
                raise TypeError('Unsupported data type.')
        else:
            if isinstance(audio_in, str):
                # for scp inputs
                if len(audio_in.split(',')) == 3:
                    data_cmd = [audio_in.split(',')]
                # for single-file inputs
                else:
                    audio_scp = generate_scp_for_sv(audio_in)
                    data_cmd = [(audio_scp, 'speech', 'sound')]
            # for raw bytes
            elif isinstance(audio_in[0], bytes):
                data_cmd = [(audio_in, 'speech', 'bytes')]
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
