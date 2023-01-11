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
from modelscope.utils.audio.audio_utils import generate_text_from_url
from modelscope.utils.constant import Frameworks, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['PunctuationProcessingPipeline']


@PIPELINES.register_module(
    Tasks.punctuation, module_name=Pipelines.punc_inference)
class PunctuationProcessingPipeline(Pipeline):
    """Punctuation Processing Inference Pipeline
    use `model` to create a Punctuation Processing pipeline.

    Args:
        model (PunctuationProcessingPipeline): A model instance, or a model local dir, or a model id in the model hub.
        kwargs (dict, `optional`):
            Extra kwargs passed into the preprocessor's constructor.
    Example:
    >>> from modelscope.pipelines import pipeline
    >>> pipeline_punc = pipeline(
    >>>    task=Tasks.punctuation, model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch')
    >>> text_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_text/punc_example.txt'
    >>> print(pipeline_punc(text_in))

    """

    def __init__(self, model: Union[Model, str] = None, **kwargs):
        """use `model` to create an asr pipeline for prediction
        """
        super().__init__(model=model, **kwargs)
        self.model_cfg = self.model.forward()

    def __call__(self, text_in: str = None) -> Dict[str, Any]:
        if len(text_in) == 0:
            raise ValueError('The input of punctuation should not be null.')
        else:
            self.text_in = text_in

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

    def forward(self, text_in: str = None) -> list:
        """Decoding
        """
        logger.info('Punctuation Processing: {0} ...'.format(text_in))
        lang = self.model_cfg['model_config']['lang']
        punc_model_path = self.model_cfg['punc_model_path']
        punc_model_config = os.path.join(
            self.model_cfg['model_workspace'],
            self.model_cfg['model_config']['punc_config'])
        mode = self.model_cfg['model_config']['mode']
        # generate text_in
        text_file = generate_text_from_url(text_in)
        data_cmd = [(text_file, 'text', 'text')]

        # generate asr inference command
        cmd = {
            'mode': mode,
            'ngpu': 1,  # 0: only CPU, ngpu>=1: gpu number if cuda is available
            'log_level': 'ERROR',
            'dtype': 'float32',
            'seed': 0,
            'name_and_type': data_cmd,
            'model_file': punc_model_path,
            'train_config': punc_model_config,
            'lang': lang
        }

        punc_result = self.run_inference(cmd)

        return punc_result

    def run_inference(self, cmd):
        punc_result = ''
        if self.framework == Frameworks.torch:
            from funasr.bin import punc_inference_launch
            punc_result = punc_inference_launch.inference_launch(
                mode=cmd['mode'],
                data_path_and_name_and_type=cmd['name_and_type'],
                ngpu=cmd['ngpu'],
                log_level=cmd['log_level'],
                dtype=cmd['dtype'],
                seed=cmd['seed'],
                train_config=cmd['train_config'],
                model_file=cmd['model_file'])
        else:
            raise ValueError('model type is mismatching')

        return punc_result
