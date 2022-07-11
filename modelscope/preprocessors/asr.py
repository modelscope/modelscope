import io
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import yaml

from modelscope.metainfo import Preprocessors
from modelscope.models.base import Model
from modelscope.utils.constant import Fields
from .base import Preprocessor
from .builder import PREPROCESSORS

__all__ = ['WavToScp']


@PREPROCESSORS.register_module(
    Fields.audio, module_name=Preprocessors.wav_to_scp)
class WavToScp(Preprocessor):
    """generate audio scp from wave or ark

    Args:
       workspace (str):
    """

    def __init__(self, workspace: str = None):
        # the workspace path
        if workspace is None or len(workspace) == 0:
            self._workspace = os.path.join(os.getcwd(), '.tmp')
        else:
            self._workspace = workspace

        if not os.path.exists(self._workspace):
            os.mkdir(self._workspace)

    def __call__(self,
                 model: List[Model] = None,
                 recog_type: str = None,
                 audio_format: str = None,
                 wav_path: str = None) -> Dict[str, Any]:
        assert len(model) > 0, 'preprocess model is invalid'
        assert len(recog_type) > 0, 'preprocess recog_type is empty'
        assert len(audio_format) > 0, 'preprocess audio_format is empty'
        assert len(wav_path) > 0, 'preprocess wav_path is empty'

        self._am_model = model[0]
        if len(model) == 2 and model[1] is not None:
            self._lm_model = model[1]
        out = self.forward(self._am_model.forward(), recog_type, audio_format,
                           wav_path)
        return out

    def forward(self, model: Dict[str, Any], recog_type: str,
                audio_format: str, wav_path: str) -> Dict[str, Any]:
        assert len(recog_type) > 0, 'preprocess recog_type is empty'
        assert len(audio_format) > 0, 'preprocess audio_format is empty'
        assert len(wav_path) > 0, 'preprocess wav_path is empty'
        assert os.path.exists(wav_path), 'preprocess wav_path does not exist'
        assert len(
            model['am_model']) > 0, 'preprocess model[am_model] is empty'
        assert len(model['am_model_path']
                   ) > 0, 'preprocess model[am_model_path] is empty'
        assert os.path.exists(
            model['am_model_path']), 'preprocess am_model_path does not exist'
        assert len(model['model_workspace']
                   ) > 0, 'preprocess model[model_workspace] is empty'
        assert os.path.exists(model['model_workspace']
                              ), 'preprocess model_workspace does not exist'
        assert len(model['model_config']
                   ) > 0, 'preprocess model[model_config] is empty'

        # the am model name
        am_model: str = model['am_model']
        # the am model file path
        am_model_path: str = model['am_model_path']
        # the recognition model dir path
        model_workspace: str = model['model_workspace']
        # the recognition model config dict
        global_model_config_dict: str = model['model_config']

        rst = {
            'workspace': os.path.join(self._workspace, recog_type),
            'am_model': am_model,
            'am_model_path': am_model_path,
            'model_workspace': model_workspace,
            # the asr type setting, eg: test dev train wav
            'recog_type': recog_type,
            # the asr audio format setting, eg: wav, kaldi_ark
            'audio_format': audio_format,
            # the test wav file path or the dataset path
            'wav_path': wav_path,
            'model_config': global_model_config_dict
        }

        out = self._config_checking(rst)
        out = self._env_setting(out)
        if audio_format == 'wav':
            out = self._scp_generation_from_wav(out)
        elif audio_format == 'kaldi_ark':
            out = self._scp_generation_from_ark(out)

        return out

    def _config_checking(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """config checking
        """

        assert inputs['model_config'].__contains__(
            'type'), 'model type does not exist'
        assert inputs['model_config'].__contains__(
            'batch_size'), 'batch_size does not exist'
        assert inputs['model_config'].__contains__(
            'am_model_config'), 'am_model_config does not exist'
        assert inputs['model_config'].__contains__(
            'asr_model_config'), 'asr_model_config does not exist'
        assert inputs['model_config'].__contains__(
            'asr_model_wav_config'), 'asr_model_wav_config does not exist'

        am_model_config: str = os.path.join(
            inputs['model_workspace'],
            inputs['model_config']['am_model_config'])
        assert os.path.exists(
            am_model_config), 'am_model_config does not exist'
        inputs['am_model_config'] = am_model_config

        asr_model_config: str = os.path.join(
            inputs['model_workspace'],
            inputs['model_config']['asr_model_config'])
        assert os.path.exists(
            asr_model_config), 'asr_model_config does not exist'

        asr_model_wav_config: str = os.path.join(
            inputs['model_workspace'],
            inputs['model_config']['asr_model_wav_config'])
        assert os.path.exists(
            asr_model_wav_config), 'asr_model_wav_config does not exist'

        inputs['model_type'] = inputs['model_config']['type']

        if inputs['audio_format'] == 'wav':
            inputs['asr_model_config'] = asr_model_wav_config
        else:
            inputs['asr_model_config'] = asr_model_config

        return inputs

    def _env_setting(self, inputs: Dict[str, Any]) -> Dict[str, Any]:

        if not os.path.exists(inputs['workspace']):
            os.mkdir(inputs['workspace'])

        inputs['output'] = os.path.join(inputs['workspace'], 'logdir')
        if not os.path.exists(inputs['output']):
            os.mkdir(inputs['output'])

        # run with datasets, should set datasets_path and text_path
        if inputs['recog_type'] != 'wav':
            inputs['datasets_path'] = inputs['wav_path']

            # run with datasets, and audio format is waveform
            if inputs['audio_format'] == 'wav':
                inputs['wav_path'] = os.path.join(inputs['datasets_path'],
                                                  'wav', inputs['recog_type'])
                inputs['hypothesis_text'] = os.path.join(
                    inputs['datasets_path'], 'transcript', 'data.text')
                assert os.path.exists(inputs['hypothesis_text']
                                      ), 'hypothesis text does not exist'

            elif inputs['audio_format'] == 'kaldi_ark':
                inputs['wav_path'] = os.path.join(inputs['datasets_path'],
                                                  inputs['recog_type'])
                inputs['hypothesis_text'] = os.path.join(
                    inputs['wav_path'], 'data.text')
                assert os.path.exists(inputs['hypothesis_text']
                                      ), 'hypothesis text does not exist'

        return inputs

    def _scp_generation_from_wav(self, inputs: Dict[str,
                                                    Any]) -> Dict[str, Any]:
        """scp generation from waveform files
        """

        # find all waveform files
        wav_list = []
        if inputs['recog_type'] == 'wav':
            file_path = inputs['wav_path']
            if os.path.isfile(file_path):
                if file_path.endswith('.wav') or file_path.endswith('.WAV'):
                    wav_list.append(file_path)
        else:
            wav_dir: str = inputs['wav_path']
            wav_list = self._recursion_dir_all_wave(wav_list, wav_dir)

        list_count: int = len(wav_list)
        inputs['wav_count'] = list_count

        # store all wav into data.0.scp
        inputs['thread_count'] = 1
        j: int = 0
        wav_list_path = os.path.join(inputs['workspace'], 'data.0.scp')
        with open(wav_list_path, 'a') as f:
            while j < list_count:
                wav_file = wav_list[j]
                wave_scp_content: str = os.path.splitext(
                    os.path.basename(wav_file))[0]
                wave_scp_content += ' ' + wav_file + '\n'
                f.write(wave_scp_content)
                j += 1

        return inputs

    def _scp_generation_from_ark(self, inputs: Dict[str,
                                                    Any]) -> Dict[str, Any]:
        """scp generation from kaldi ark file
        """

        inputs['thread_count'] = 1
        ark_scp_path = os.path.join(inputs['wav_path'], 'data.scp')
        ark_file_path = os.path.join(inputs['wav_path'], 'data.ark')
        assert os.path.exists(ark_scp_path), 'data.scp does not exist'
        assert os.path.exists(ark_file_path), 'data.ark does not exist'

        new_ark_scp_path = os.path.join(inputs['workspace'], 'data.0.scp')

        with open(ark_scp_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        with open(new_ark_scp_path, 'w', encoding='utf-8') as n:
            for line in lines:
                outs = line.strip().split(' ')
                if len(outs) == 2:
                    key = outs[0]
                    sub = outs[1].split(':')
                    if len(sub) == 2:
                        nums = sub[1]
                        content = key + ' ' + ark_file_path + ':' + nums + '\n'
                        n.write(content)

        return inputs

    def _recursion_dir_all_wave(self, wav_list,
                                dir_path: str) -> Dict[str, Any]:
        dir_files = os.listdir(dir_path)
        for file in dir_files:
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                if file_path.endswith('.wav') or file_path.endswith('.WAV'):
                    wav_list.append(file_path)
            elif os.path.isdir(file_path):
                self._recursion_dir_all_wave(wav_list, file_path)

        return wav_list
