import os
import shutil
import threading
from typing import Any, Dict, List, Sequence, Tuple, Union

import yaml

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import WavToScp
from modelscope.utils.constant import Tasks
from .asr_engine.common import asr_utils

__all__ = ['AutomaticSpeechRecognitionPipeline']


@PIPELINES.register_module(
    Tasks.auto_speech_recognition, module_name=Pipelines.asr_inference)
class AutomaticSpeechRecognitionPipeline(Pipeline):
    """ASR Pipeline
    """

    def __init__(self,
                 model: Union[List[Model], List[str]] = None,
                 preprocessor: WavToScp = None,
                 **kwargs):
        """use `model` and `preprocessor` to create an asr pipeline for prediction
        """
        from .asr_engine import asr_env_checking
        assert model is not None, 'asr model should be provided'

        model_list: List = []
        if isinstance(model[0], Model):
            model_list = model
        else:
            model_list.append(Model.from_pretrained(model[0]))
            if len(model) == 2 and model[1] is not None:
                model_list.append(Model.from_pretrained(model[1]))

        super().__init__(model=model_list, preprocessor=preprocessor, **kwargs)

        self._preprocessor = preprocessor
        self._am_model = model_list[0]
        if len(model_list) == 2 and model_list[1] is not None:
            self._lm_model = model_list[1]

    def __call__(self,
                 wav_path: str,
                 recog_type: str = None,
                 audio_format: str = None,
                 workspace: str = None) -> Dict[str, Any]:
        assert len(wav_path) > 0, 'wav_path should be provided'

        self._recog_type = recog_type
        self._audio_format = audio_format
        self._workspace = workspace
        self._wav_path = wav_path

        if recog_type is None or audio_format is None or workspace is None:
            self._recog_type, self._audio_format, self._workspace, self._wav_path = asr_utils.type_checking(
                wav_path, recog_type, audio_format, workspace)

        if self._preprocessor is None:
            self._preprocessor = WavToScp(workspace=self._workspace)

        output = self._preprocessor.forward(self._am_model.forward(),
                                            self._recog_type,
                                            self._audio_format, self._wav_path)
        output = self.forward(output)
        rst = self.postprocess(output)
        return rst

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Decoding
        """

        j: int = 0
        process = []

        while j < inputs['thread_count']:
            data_cmd: Sequence[Tuple[str, str, str]]
            if inputs['audio_format'] == 'wav':
                data_cmd = [(os.path.join(inputs['workspace'],
                                          'data.' + str(j) + '.scp'), 'speech',
                             'sound')]
            elif inputs['audio_format'] == 'kaldi_ark':
                data_cmd = [(os.path.join(inputs['workspace'],
                                          'data.' + str(j) + '.scp'), 'speech',
                             'kaldi_ark')]

            output_dir: str = os.path.join(inputs['output'],
                                           'output.' + str(j))
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            config_file = open(inputs['asr_model_config'])
            root = yaml.full_load(config_file)
            config_file.close()
            frontend_conf = None
            if 'frontend_conf' in root:
                frontend_conf = root['frontend_conf']

            cmd = {
                'model_type': inputs['model_type'],
                'beam_size': root['beam_size'],
                'penalty': root['penalty'],
                'maxlenratio': root['maxlenratio'],
                'minlenratio': root['minlenratio'],
                'ctc_weight': root['ctc_weight'],
                'lm_weight': root['lm_weight'],
                'output_dir': output_dir,
                'ngpu': 0,
                'log_level': 'ERROR',
                'data_path_and_name_and_type': data_cmd,
                'asr_train_config': inputs['am_model_config'],
                'asr_model_file': inputs['am_model_path'],
                'batch_size': inputs['model_config']['batch_size'],
                'frontend_conf': frontend_conf
            }

            thread = AsrInferenceThread(j, cmd)
            thread.start()
            j += 1
            process.append(thread)

        for p in process:
            p.join()

        return inputs

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """process the asr results
        """

        rst = {'rec_result': 'None'}

        # single wav task
        if inputs['recog_type'] == 'wav' and inputs['audio_format'] == 'wav':
            text_file: str = os.path.join(inputs['output'], 'output.0',
                                          '1best_recog', 'text')

            if os.path.exists(text_file):
                f = open(text_file, 'r')
                result_str: str = f.readline()
                f.close()
                if len(result_str) > 0:
                    result_list = result_str.split()
                    if len(result_list) >= 2:
                        rst['rec_result'] = result_list[1]

        # run with datasets, and audio format is waveform or kaldi_ark
        elif inputs['recog_type'] != 'wav':
            inputs['reference_text'] = self._ref_text_tidy(inputs)
            inputs['datasets_result'] = asr_utils.compute_wer(
                inputs['hypothesis_text'], inputs['reference_text'])

        else:
            raise ValueError('recog_type and audio_format are mismatching')

        if 'datasets_result' in inputs:
            rst['datasets_result'] = inputs['datasets_result']

        # remove workspace dir (.tmp)
        if os.path.exists(self._workspace):
            shutil.rmtree(self._workspace)

        return rst

    def _ref_text_tidy(self, inputs: Dict[str, Any]) -> str:
        ref_text: str = os.path.join(inputs['output'], 'text.ref')
        k: int = 0

        while k < inputs['thread_count']:
            output_text = os.path.join(inputs['output'], 'output.' + str(k),
                                       '1best_recog', 'text')
            if os.path.exists(output_text):
                with open(output_text, 'r', encoding='utf-8') as i:
                    lines = i.readlines()

                with open(ref_text, 'a', encoding='utf-8') as o:
                    for line in lines:
                        o.write(line)

            k += 1

        return ref_text


class AsrInferenceThread(threading.Thread):

    def __init__(self, threadID, cmd):
        threading.Thread.__init__(self)
        self._threadID = threadID
        self._cmd = cmd

    def run(self):
        if self._cmd['model_type'] == 'pytorch':
            from .asr_engine import asr_inference_paraformer_espnet
            asr_inference_paraformer_espnet.asr_inference(
                batch_size=self._cmd['batch_size'],
                output_dir=self._cmd['output_dir'],
                maxlenratio=self._cmd['maxlenratio'],
                minlenratio=self._cmd['minlenratio'],
                beam_size=self._cmd['beam_size'],
                ngpu=self._cmd['ngpu'],
                ctc_weight=self._cmd['ctc_weight'],
                lm_weight=self._cmd['lm_weight'],
                penalty=self._cmd['penalty'],
                log_level=self._cmd['log_level'],
                data_path_and_name_and_type=self.
                _cmd['data_path_and_name_and_type'],
                asr_train_config=self._cmd['asr_train_config'],
                asr_model_file=self._cmd['asr_model_file'],
                frontend_conf=self._cmd['frontend_conf'])
