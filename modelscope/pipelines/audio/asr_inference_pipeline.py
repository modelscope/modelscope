import os
from typing import Any, Dict, List, Sequence, Tuple, Union

import yaml

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import WavToScp
from modelscope.utils.constant import Frameworks, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['AutomaticSpeechRecognitionPipeline']


@PIPELINES.register_module(
    Tasks.auto_speech_recognition, module_name=Pipelines.asr_inference)
class AutomaticSpeechRecognitionPipeline(Pipeline):
    """ASR Inference Pipeline
    """

    def __init__(self,
                 model: Union[Model, str] = None,
                 preprocessor: WavToScp = None,
                 **kwargs):
        """use `model` and `preprocessor` to create an asr pipeline for prediction
        """
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)

    def __call__(self,
                 audio_in: Union[str, bytes],
                 recog_type: str = None,
                 audio_format: str = None) -> Dict[str, Any]:
        from easyasr.common import asr_utils

        self.recog_type = recog_type
        self.audio_format = audio_format
        self.audio_in = audio_in

        if recog_type is None or audio_format is None:
            self.recog_type, self.audio_format, self.audio_in = asr_utils.type_checking(
                audio_in, recog_type, audio_format)

        if self.preprocessor is None:
            self.preprocessor = WavToScp()

        output = self.preprocessor.forward(self.model.forward(),
                                           self.recog_type, self.audio_format,
                                           self.audio_in)
        output = self.forward(output)
        rst = self.postprocess(output)
        return rst

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Decoding
        """

        logger.info(f"Decoding with {inputs['audio_format']} files ...")

        data_cmd: Sequence[Tuple[str, str]]
        if inputs['audio_format'] == 'wav' or inputs['audio_format'] == 'pcm':
            data_cmd = ['speech', 'sound']
        elif inputs['audio_format'] == 'kaldi_ark':
            data_cmd = ['speech', 'kaldi_ark']
        elif inputs['audio_format'] == 'tfrecord':
            data_cmd = ['speech', 'tfrecord']

        # generate asr inference command
        cmd = {
            'model_type': inputs['model_type'],
            'ngpu': 1,  # 0: only CPU, ngpu>=1: gpu number if cuda is available
            'log_level': 'ERROR',
            'audio_in': inputs['audio_lists'],
            'name_and_type': data_cmd,
            'asr_model_file': inputs['am_model_path'],
            'idx_text': ''
        }

        if self.framework == Frameworks.torch:
            config_file = open(inputs['asr_model_config'])
            root = yaml.full_load(config_file)
            config_file.close()
            frontend_conf = None
            if 'frontend_conf' in root:
                frontend_conf = root['frontend_conf']

            cmd['beam_size'] = root['beam_size']
            cmd['penalty'] = root['penalty']
            cmd['maxlenratio'] = root['maxlenratio']
            cmd['minlenratio'] = root['minlenratio']
            cmd['ctc_weight'] = root['ctc_weight']
            cmd['lm_weight'] = root['lm_weight']
            cmd['asr_train_config'] = inputs['am_model_config']
            cmd['batch_size'] = inputs['model_config']['batch_size']
            cmd['frontend_conf'] = frontend_conf

        elif self.framework == Frameworks.tf:
            cmd['fs'] = inputs['model_config']['fs']
            cmd['hop_length'] = inputs['model_config']['hop_length']
            cmd['feature_dims'] = inputs['model_config']['feature_dims']
            cmd['predictions_file'] = 'text'
            cmd['mvn_file'] = inputs['am_mvn_file']
            cmd['vocab_file'] = inputs['vocab_file']
            if 'idx_text' in inputs:
                cmd['idx_text'] = inputs['idx_text']

        else:
            raise ValueError('model type is mismatching')

        inputs['asr_result'] = self.run_inference(cmd)

        return inputs

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """process the asr results
        """
        from easyasr.common import asr_utils

        logger.info('Computing the result of ASR ...')

        rst = {}

        # single wav or pcm task
        if inputs['recog_type'] == 'wav':
            if 'asr_result' in inputs and len(inputs['asr_result']) > 0:
                text = inputs['asr_result'][0]['value']
                if len(text) > 0:
                    rst[OutputKeys.TEXT] = text

        # run with datasets, and audio format is waveform or kaldi_ark or tfrecord
        elif inputs['recog_type'] != 'wav':
            inputs['reference_list'] = self.ref_list_tidy(inputs)
            inputs['datasets_result'] = asr_utils.compute_wer(
                inputs['asr_result'], inputs['reference_list'])

        else:
            raise ValueError('recog_type and audio_format are mismatching')

        if 'datasets_result' in inputs:
            rst[OutputKeys.TEXT] = inputs['datasets_result']

        return rst

    def ref_list_tidy(self, inputs: Dict[str, Any]) -> List[Any]:
        ref_list = []

        if inputs['audio_format'] == 'tfrecord':
            # should assemble idx + txt
            with open(inputs['reference_text'], 'r', encoding='utf-8') as r:
                text_lines = r.readlines()

            with open(inputs['idx_text'], 'r', encoding='utf-8') as i:
                idx_lines = i.readlines()

            j: int = 0
            while j < min(len(text_lines), len(idx_lines)):
                idx_str = idx_lines[j].strip()
                text_str = text_lines[j].strip().replace(' ', '')
                item = {'key': idx_str, 'value': text_str}
                ref_list.append(item)
                j += 1

        else:
            # text contain idx + sentence
            with open(inputs['reference_text'], 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                line_item = line.split()
                item = {'key': line_item[0], 'value': line_item[1]}
                ref_list.append(item)

        return ref_list

    def run_inference(self, cmd):
        asr_result = []
        if self.framework == Frameworks.torch:
            from easyasr import asr_inference_paraformer_espnet
            asr_result = asr_inference_paraformer_espnet.asr_inference(
                batch_size=cmd['batch_size'],
                maxlenratio=cmd['maxlenratio'],
                minlenratio=cmd['minlenratio'],
                beam_size=cmd['beam_size'],
                ngpu=cmd['ngpu'],
                ctc_weight=cmd['ctc_weight'],
                lm_weight=cmd['lm_weight'],
                penalty=cmd['penalty'],
                log_level=cmd['log_level'],
                name_and_type=cmd['name_and_type'],
                audio_lists=cmd['audio_in'],
                asr_train_config=cmd['asr_train_config'],
                asr_model_file=cmd['asr_model_file'],
                frontend_conf=cmd['frontend_conf'])
        elif self.framework == Frameworks.tf:
            from easyasr import asr_inference_paraformer_tf
            asr_result = asr_inference_paraformer_tf.asr_inference(
                ngpu=cmd['ngpu'],
                name_and_type=cmd['name_and_type'],
                audio_lists=cmd['audio_in'],
                idx_text_file=cmd['idx_text'],
                asr_model_file=cmd['asr_model_file'],
                vocab_file=cmd['vocab_file'],
                am_mvn_file=cmd['mvn_file'],
                predictions_file=cmd['predictions_file'],
                fs=cmd['fs'],
                hop_length=cmd['hop_length'],
                feature_dims=cmd['feature_dims'])

        return asr_result
