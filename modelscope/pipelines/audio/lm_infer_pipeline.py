# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, Union

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.audio.audio_utils import generate_text_from_url
from modelscope.utils.config import Config
from modelscope.utils.constant import Frameworks, ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['LanguageModelPipeline']


@PIPELINES.register_module(
    Tasks.language_model, module_name=Pipelines.lm_inference)
class LanguageModelPipeline(Pipeline):
    """Language Model Inference Pipeline

    Example:
    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.utils.constant import Tasks

    >>> pipeline_lm = pipeline(
    >>>    task=Tasks.language_model,
    >>>    model='damo/speech_transformer_lm_zh-cn-common-vocab8404-pytorch')
    >>> text_in='hello 大 家 好 呀'
    >>> print(pipeline_lm(text_in))

    """

    def __init__(self, model: Union[Model, str] = None, **kwargs):
        """
        Use `model` to create a LM pipeline for prediction
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
            model_file('str'):
                LM model file
            train_config('str'):
                LM infer configuration
            num_workers('int'):
                the number of workers used for DataLoader
            log_level('str'):
                log level
            log_base('float', defaults to 10.0):
                the base of logarithm for Perplexity
            split_with_space('bool'):
                split the input sentence by space
            seg_dict_file('str'):
                seg dict file
            param_dict('dict'):
                extra kwargs
        """
        super().__init__(model=model, **kwargs)
        config_path = os.path.join(model, ModelFile.CONFIGURATION)
        self.cmd = self.get_cmd(config_path, kwargs)

        from funasr.bin import lm_inference_launch
        self.funasr_infer_modelscope = lm_inference_launch.inference_launch(
            mode=self.cmd['mode'],
            batch_size=self.cmd['batch_size'],
            dtype=self.cmd['dtype'],
            ngpu=self.cmd['ngpu'],
            seed=self.cmd['seed'],
            num_workers=self.cmd['num_workers'],
            log_level=self.cmd['log_level'],
            key_file=self.cmd['key_file'],
            train_config=self.cmd['train_config'],
            model_file=self.cmd['model_file'],
            log_base=self.cmd['log_base'],
            split_with_space=self.cmd['split_with_space'],
            seg_dict_file=self.cmd['seg_dict_file'],
            output_dir=self.cmd['output_dir'],
            param_dict=self.cmd['param_dict'])

    def __call__(self,
                 text_in: str = None,
                 output_dir: str = None,
                 param_dict: dict = None) -> Dict[str, Any]:
        """
        Compute PPL
        Args:
            text_in('str'):
                - A text str input
                - A local text file input endswith .txt or .scp
                - A url text file input
            output_dir('str'):
                output dir
            param_dict('dict'):
                extra kwargs
        Return:
            A dictionary of result or a list of dictionary of result.

            The dictionary contain the following keys:
            - **text** ('str') --The PPL result.
        """
        if len(text_in) == 0:
            raise ValueError('The input of lm should not be null.')
        else:
            self.text_in = text_in
        if output_dir is not None:
            self.cmd['output_dir'] = output_dir
        if param_dict is not None:
            self.cmd['param_dict'] = param_dict

        output = self.forward(self.text_in)
        result = self.postprocess(output)
        return result

    def postprocess(self, inputs: list) -> Dict[str, Any]:
        """Postprocessing
        """
        rst = {}
        for i in range(len(inputs)):
            if i == 0:
                text = inputs[0]['value']
                if len(text) > 0:
                    rst[OutputKeys.TEXT] = text
            else:
                rst[inputs[i]['key']] = inputs[i]['value']
        return rst

    def get_cmd(self, config_path, extra_args) -> Dict[str, Any]:
        # generate inference command
        model_cfg = Config.from_file(config_path)
        model_dir = os.path.dirname(config_path)
        mode = model_cfg.model['model_config']['mode']
        lm_model_path = os.path.join(
            model_dir, model_cfg.model['model_config']['lm_model_name'])
        lm_model_config = os.path.join(
            model_dir, model_cfg.model['model_config']['lm_model_config'])
        seg_dict_file = None
        if 'seg_dict_file' in model_cfg.model['model_config']:
            seg_dict_file = os.path.join(
                model_dir, model_cfg.model['model_config']['seg_dict_file'])

        cmd = {
            'mode': mode,
            'batch_size': 1,
            'dtype': 'float32',
            'ngpu': 1,  # 0: only CPU, ngpu>=1: gpu number if cuda is available
            'seed': 0,
            'num_workers': 0,
            'log_level': 'ERROR',
            'key_file': None,
            'train_config': lm_model_config,
            'model_file': lm_model_path,
            'log_base': 10.0,
            'allow_variable_data_keys': False,
            'split_with_space': True,
            'seg_dict_file': seg_dict_file,
            'output_dir': None,
            'param_dict': None,
        }

        user_args_dict = [
            'batch_size',
            'ngpu',
            'num_workers',
            'log_level',
            'train_config',
            'model_file',
            'log_base',
            'split_with_space',
            'seg_dict_file',
            'output_dir',
            'param_dict',
        ]

        for user_args in user_args_dict:
            if user_args in extra_args and extra_args[user_args] is not None:
                cmd[user_args] = extra_args[user_args]

        return cmd

    def forward(self, text_in: str = None) -> list:
        """Decoding
        """
        logger.info('Compute PPL : {0} ...'.format(text_in))
        # generate text_in
        text_file, raw_inputs = generate_text_from_url(text_in)
        data_cmd = None
        if raw_inputs is None:
            data_cmd = [(text_file, 'text', 'text')]
        elif text_file is None and raw_inputs is not None:
            data_cmd = None

        self.cmd['name_and_type'] = data_cmd
        self.cmd['raw_inputs'] = raw_inputs
        lm_result = self.run_inference(self.cmd)

        return lm_result

    def run_inference(self, cmd):
        if self.framework == Frameworks.torch:
            lm_result = self.funasr_infer_modelscope(
                data_path_and_name_and_type=cmd['name_and_type'],
                raw_inputs=cmd['raw_inputs'],
                output_dir_v2=cmd['output_dir'],
                param_dict=cmd['param_dict'])
        else:
            raise ValueError('model type is mismatching')

        return lm_result
