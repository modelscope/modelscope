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
from modelscope.utils.audio.audio_utils import generate_sv_scp_from_url
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
    >>> pipeline_punc = pipeline(
    >>>    task=Tasks.speaker_verification, model='damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch')
    >>> audio_in=('','')
    >>> print(pipeline_punc(audio_in))

    """

    def __init__(self, model: Union[Model, str] = None, **kwargs):
        """use `model` to create an asr pipeline for prediction
        """
        super().__init__(model=model, **kwargs)
        self.model_cfg = self.model.forward()

    def __call__(self, audio_in: tuple = None) -> Dict[str, Any]:
        if len(audio_in) == 0:
            raise ValueError('The input of ITN should not be null.')
        else:
            self.audio_in = audio_in

        output = self.forward(self.audio_in)
        result = self.postprocess(output)
        return result

    def postprocess(self, inputs: list) -> Dict[str, Any]:
        """Postprocessing
        """
        rst = {}
        for i in range(len(inputs)):
            if i == 0:
                score = inputs[0]['value']
                rst[OutputKeys.SCORES] = score
            else:
                rst[inputs[i]['key']] = inputs[i]['value']
        return rst

    def forward(self, audio_in: tuple = None) -> list:
        """Decoding
        """
        logger.info(
            'Speaker Verification Processing: {0} ...'.format(audio_in))
        sv_model_path = self.model_cfg['sv_model_path']
        sv_model_config = os.path.join(
            self.model_cfg['model_workspace'],
            self.model_cfg['model_config']['sv_model_config'])
        mode = self.model_cfg['model_config']['mode']
        # generate audio_scp
        audio_scp_1, audio_scp_2 = generate_sv_scp_from_url(audio_in)
        data_cmd = [(audio_scp_1, 'speech', 'sound'),
                    (audio_scp_2, 'ref_speech', 'sound')]

        # generate asr inference command
        cmd = {
            'mode': mode,
            'output_dir': None,
            'ngpu': 1,  # 0: only CPU, ngpu>=1: gpu number if cuda is available
            'batch_size': 1,
            'num_workers': 1,
            'log_level': 'ERROR',
            'dtype': 'float32',
            'seed': 0,
            'key_file': None,
            'name_and_type': data_cmd,
            'sv_model_file': sv_model_path,
            'sv_train_config': sv_model_config,
            'model_tag': None
        }

        punc_result = self.run_inference(cmd)

        return punc_result

    def run_inference(self, cmd):
        sv_result = ''
        if self.framework == Frameworks.torch:
            from funasr.bin import sv_inference_launch
            sv_result = sv_inference_launch.inference_launch(
                mode=cmd['mode'],
                data_path_and_name_and_type=cmd['name_and_type'],
                ngpu=cmd['ngpu'],
                log_level=cmd['log_level'],
                dtype=cmd['dtype'],
                seed=cmd['seed'],
                sv_train_config=cmd['sv_train_config'],
                sv_model_file=cmd['sv_model_file'],
                output_dir=cmd['output_dir'],
                batch_size=cmd['batch_size'],
                num_workers=cmd['num_workers'],
                key_file=cmd['key_file'],
                model_tag=cmd['model_tag'])
        else:
            raise ValueError('model type is mismatching')

        return sv_result
