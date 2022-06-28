import os
import shutil
import stat
from pathlib import Path
from typing import Any, Dict, List

import yaml

from modelscope.metainfo import Preprocessors
from modelscope.models.base import Model
from modelscope.utils.constant import Fields
from .base import Preprocessor
from .builder import PREPROCESSORS

__all__ = ['WavToLists']


@PREPROCESSORS.register_module(
    Fields.audio, module_name=Preprocessors.wav_to_lists)
class WavToLists(Preprocessor):
    """generate audio lists file from wav

    Args:
        workspace (str): store temporarily kws intermedium and result
    """

    def __init__(self, workspace: str = None):
        # the workspace path
        if len(workspace) == 0:
            self._workspace = os.path.join(os.getcwd(), '.tmp')
        else:
            self._workspace = workspace

        if not os.path.exists(self._workspace):
            os.mkdir(self._workspace)

    def __call__(self,
                 model: Model = None,
                 kws_type: str = None,
                 wav_path: List[str] = None) -> Dict[str, Any]:
        """Call functions to load model and wav.

        Args:
            model (Model): model should be provided
            kws_type (str): kws work type: wav, neg_testsets, pos_testsets, roc
            wav_path (List[str]): wav_path[0] is positive wav path,  wav_path[1] is negative wav path
        Returns:
            Dict[str, Any]: the kws result
        """

        assert model is not None, 'preprocess kws model should be provided'
        assert kws_type in ['wav', 'pos_testsets', 'neg_testsets', 'roc'
                            ], f'preprocess kws_type {kws_type} is invalid'
        assert wav_path[0] is not None or wav_path[
            1] is not None, 'preprocess wav_path is invalid'

        self._model = model
        out = self.forward(self._model.forward(), kws_type, wav_path)
        return out

    def forward(self, model: Dict[str, Any], kws_type: str,
                wav_path: List[str]) -> Dict[str, Any]:
        assert len(kws_type) > 0, 'preprocess kws_type is empty'
        assert len(
            model['config_path']) > 0, 'preprocess model[config_path] is empty'
        assert os.path.exists(
            model['config_path']), 'model config.yaml is absent'

        inputs = model.copy()

        inputs['kws_set'] = kws_type
        inputs['workspace'] = self._workspace
        if wav_path[0] is not None:
            inputs['pos_wav_path'] = wav_path[0]
        if wav_path[1] is not None:
            inputs['neg_wav_path'] = wav_path[1]

        out = self._read_config(inputs)
        out = self._generate_wav_lists(out)

        return out

    def _read_config(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """read and parse config.yaml to get all model files
        """

        assert os.path.exists(
            inputs['config_path']), 'model config yaml file does not exist'

        config_file = open(inputs['config_path'])
        root = yaml.full_load(config_file)
        config_file.close()

        inputs['cfg_file'] = root['cfg_file']
        inputs['cfg_file_path'] = os.path.join(inputs['model_workspace'],
                                               root['cfg_file'])
        inputs['keyword_grammar'] = root['keyword_grammar']
        inputs['keyword_grammar_path'] = os.path.join(
            inputs['model_workspace'], root['keyword_grammar'])
        inputs['sample_rate'] = str(root['sample_rate'])
        inputs['kws_tool'] = root['kws_tool']

        if os.path.exists(
                os.path.join(inputs['workspace'], inputs['kws_tool'])):
            inputs['kws_tool_path'] = os.path.join(inputs['workspace'],
                                                   inputs['kws_tool'])
        elif os.path.exists(os.path.join('/usr/bin', inputs['kws_tool'])):
            inputs['kws_tool_path'] = os.path.join('/usr/bin',
                                                   inputs['kws_tool'])
        elif os.path.exists(os.path.join('/bin', inputs['kws_tool'])):
            inputs['kws_tool_path'] = os.path.join('/bin', inputs['kws_tool'])

        assert os.path.exists(inputs['kws_tool_path']), 'cannot find kwsbp'
        os.chmod(inputs['kws_tool_path'],
                 stat.S_IXUSR + stat.S_IXGRP + stat.S_IXOTH)

        self._config_checking(inputs)
        return inputs

    def _generate_wav_lists(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """assemble wav lists
        """

        if inputs['kws_set'] == 'wav':
            inputs['pos_num_thread'] = 1
            wave_scp_content: str = inputs['pos_wav_path'] + '\n'

            with open(os.path.join(inputs['pos_data_path'], 'wave.list'),
                      'a') as f:
                f.write(wave_scp_content)

            inputs['pos_wav_count'] = 1

        if inputs['kws_set'] in ['pos_testsets', 'roc']:
            # find all positive wave
            wav_list = []
            wav_dir = inputs['pos_wav_path']
            wav_list = self._recursion_dir_all_wave(wav_list, wav_dir)

            list_count: int = len(wav_list)
            inputs['pos_wav_count'] = list_count

            if list_count <= 128:
                inputs['pos_num_thread'] = list_count
                j: int = 0
                while j < list_count:
                    wave_scp_content: str = wav_list[j] + '\n'
                    wav_list_path = inputs['pos_data_path'] + '/wave.' + str(
                        j) + '.list'
                    with open(wav_list_path, 'a') as f:
                        f.write(wave_scp_content)
                    j += 1

            else:
                inputs['pos_num_thread'] = 128
                j: int = 0
                k: int = 0
                while j < list_count:
                    wave_scp_content: str = wav_list[j] + '\n'
                    wav_list_path = inputs['pos_data_path'] + '/wave.' + str(
                        k) + '.list'
                    with open(wav_list_path, 'a') as f:
                        f.write(wave_scp_content)
                    j += 1
                    k += 1
                    if k >= 128:
                        k = 0

        if inputs['kws_set'] in ['neg_testsets', 'roc']:
            # find all negative wave
            wav_list = []
            wav_dir = inputs['neg_wav_path']
            wav_list = self._recursion_dir_all_wave(wav_list, wav_dir)

            list_count: int = len(wav_list)
            inputs['neg_wav_count'] = list_count

            if list_count <= 128:
                inputs['neg_num_thread'] = list_count
                j: int = 0
                while j < list_count:
                    wave_scp_content: str = wav_list[j] + '\n'
                    wav_list_path = inputs['neg_data_path'] + '/wave.' + str(
                        j) + '.list'
                    with open(wav_list_path, 'a') as f:
                        f.write(wave_scp_content)
                    j += 1

            else:
                inputs['neg_num_thread'] = 128
                j: int = 0
                k: int = 0
                while j < list_count:
                    wave_scp_content: str = wav_list[j] + '\n'
                    wav_list_path = inputs['neg_data_path'] + '/wave.' + str(
                        k) + '.list'
                    with open(wav_list_path, 'a') as f:
                        f.write(wave_scp_content)
                    j += 1
                    k += 1
                    if k >= 128:
                        k = 0

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

    def _config_checking(self, inputs: Dict[str, Any]):

        if inputs['kws_set'] in ['wav', 'pos_testsets', 'roc']:
            inputs['pos_data_path'] = os.path.join(inputs['workspace'],
                                                   'pos_data')
            if not os.path.exists(inputs['pos_data_path']):
                os.mkdir(inputs['pos_data_path'])
            else:
                shutil.rmtree(inputs['pos_data_path'])
                os.mkdir(inputs['pos_data_path'])

            inputs['pos_dump_path'] = os.path.join(inputs['workspace'],
                                                   'pos_dump')
            if not os.path.exists(inputs['pos_dump_path']):
                os.mkdir(inputs['pos_dump_path'])
            else:
                shutil.rmtree(inputs['pos_dump_path'])
                os.mkdir(inputs['pos_dump_path'])

        if inputs['kws_set'] in ['neg_testsets', 'roc']:
            inputs['neg_data_path'] = os.path.join(inputs['workspace'],
                                                   'neg_data')
            if not os.path.exists(inputs['neg_data_path']):
                os.mkdir(inputs['neg_data_path'])
            else:
                shutil.rmtree(inputs['neg_data_path'])
                os.mkdir(inputs['neg_data_path'])

            inputs['neg_dump_path'] = os.path.join(inputs['workspace'],
                                                   'neg_dump')
            if not os.path.exists(inputs['neg_dump_path']):
                os.mkdir(inputs['neg_dump_path'])
            else:
                shutil.rmtree(inputs['neg_dump_path'])
                os.mkdir(inputs['neg_dump_path'])
