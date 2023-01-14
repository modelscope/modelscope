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
from modelscope.utils.constant import Frameworks, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['InverseTextProcessingPipeline']


@PIPELINES.register_module(
    Tasks.inverse_text_processing, module_name=Pipelines.itn_inference)
class InverseTextProcessingPipeline(Pipeline):
    """Inverse Text Processing Inference Pipeline
    use `model` to create a Inverse Text Processing pipeline.

    Args:
        model (BartForTextErrorCorrection): A model instance, or a model local dir, or a model id in the model hub.
        kwargs (dict, `optional`):
            Extra kwargs passed into the preprocessor's constructor.
    Example:
    >>> from modelscope.pipelines import pipeline
    >>> pipeline_itn = pipeline(
    >>>    task=Tasks.inverse_text_processing, model='damo/speech_inverse_text_processing_fun-text-processing-itn-id')
    >>> sentence = 'sembilan ribu sembilan ratus sembilan puluh sembilan'
    >>> print(pipeline_itn(sentence))

    To view other examples plese check tests/pipelines/test_inverse_text_processing.py.
    """

    def __init__(self, model: Union[Model, str] = None, **kwargs):
        """use `model` to create an asr pipeline for prediction
        """
        super().__init__(model=model, **kwargs)
        self.model_cfg = self.model.forward()

    def __call__(self, text_in: str = None) -> Dict[str, Any]:

        if len(text_in) == 0:
            raise ValueError('The input of ITN should not be null.')
        else:
            self.text_in = text_in
        output = {}
        itn_result = self.forward(self.text_in)
        output['text'] = itn_result

        return output

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocessing
        """
        return inputs

    def forward(self, text_in: str = None) -> str:
        """Decoding
        """
        logger.info('Inverse Text Normalization: {0} ...'.format(text_in))
        lang = self.model_cfg['model_config']['lang']
        model_dir = self.model_cfg['model_workspace']
        itn_model_path = self.model_cfg['itn_model_path']

        # make directory recursively
        cache_dir = os.path.join(model_dir, lang, '.cache')
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, mode=0o777, exist_ok=True)

        name = '_{0}_itn.far'.format(lang)
        far_file = os.path.join(cache_dir, name)

        # copy file into cache_dir
        shutil.copy(itn_model_path, far_file)

        # generate itn inference command
        cmd = {
            'ngpu': 0,
            'log_level': 'ERROR',
            'text_in': text_in,
            'itn_model_file': far_file,
            'cache_dir': cache_dir,
            'overwrite_cache': False,
            'enable_standalone_number': True,
            'enable_0_to_9': True,
            'lang': lang,
            'verbose': False,
        }

        itn_result = self.run_inference(cmd)

        return itn_result

    def run_inference(self, cmd):
        itn_result = ''
        if self.framework == Frameworks.torch:
            from fun_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
            if cmd['lang'] == 'ja':
                itn_normalizer = InverseNormalizer(
                    lang=cmd['lang'],
                    cache_dir=cmd['cache_dir'],
                    overwrite_cache=cmd['overwrite_cache'],
                    enable_standalone_number=cmd['enable_standalone_number'],
                    enable_0_to_9=cmd['enable_0_to_9'])
            else:
                itn_normalizer = InverseNormalizer(
                    lang=cmd['lang'],
                    cache_dir=cmd['cache_dir'],
                    overwrite_cache=cmd['overwrite_cache'])
            itn_result = itn_normalizer.inverse_normalize(
                cmd['text_in'], verbose=cmd['verbose'])

        else:
            raise ValueError('model type is mismatching')

        return itn_result
