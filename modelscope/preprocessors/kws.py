import os
from typing import Any, Dict, List, Union

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
    """

    def __init__(self):
        pass

    def __call__(self, model: Model, wav_path: Union[List[str],
                                                     str]) -> Dict[str, Any]:
        """Call functions to load model and wav.

        Args:
            model (Model): model should be provided
            wav_path (Union[List[str], str]): wav_path[0] is positive wav path,  wav_path[1] is negative wav path
        Returns:
            Dict[str, Any]: the kws result
        """

        self.model = model
        out = self.forward(self.model.forward(), wav_path)
        return out

    def forward(self, model: Dict[str, Any],
                wav_path: Union[List[str], str]) -> Dict[str, Any]:
        assert len(
            model['config_path']) > 0, 'preprocess model[config_path] is empty'
        assert os.path.exists(
            model['config_path']), 'model config.yaml is absent'

        inputs = model.copy()

        wav_list = [None, None]
        if isinstance(wav_path, str):
            wav_list[0] = wav_path
        else:
            wav_list = wav_path

        import kws_util.common
        kws_type = kws_util.common.type_checking(wav_list)
        assert kws_type in ['wav', 'pos_testsets', 'neg_testsets', 'roc'
                            ], f'preprocess kws_type {kws_type} is invalid'

        inputs['kws_set'] = kws_type
        if wav_list[0] is not None:
            inputs['pos_wav_path'] = wav_list[0]
        if wav_list[1] is not None:
            inputs['neg_wav_path'] = wav_list[1]

        out = self.read_config(inputs)
        out = self.generate_wav_lists(out)

        return out

    def read_config(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
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
        inputs['sample_rate'] = root['sample_rate']

        return inputs

    def generate_wav_lists(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """assemble wav lists
        """
        import kws_util.common

        if inputs['kws_set'] == 'wav':
            wav_list = []
            wave_scp_content: str = inputs['pos_wav_path']
            wav_list.append(wave_scp_content)
            inputs['pos_wav_list'] = wav_list
            inputs['pos_wav_count'] = 1
            inputs['pos_num_thread'] = 1

        if inputs['kws_set'] in ['pos_testsets', 'roc']:
            # find all positive wave
            wav_list = []
            wav_dir = inputs['pos_wav_path']
            wav_list = kws_util.common.recursion_dir_all_wav(wav_list, wav_dir)
            inputs['pos_wav_list'] = wav_list

            list_count: int = len(wav_list)
            inputs['pos_wav_count'] = list_count

            if list_count <= 128:
                inputs['pos_num_thread'] = list_count
            else:
                inputs['pos_num_thread'] = 128

        if inputs['kws_set'] in ['neg_testsets', 'roc']:
            # find all negative wave
            wav_list = []
            wav_dir = inputs['neg_wav_path']
            wav_list = kws_util.common.recursion_dir_all_wav(wav_list, wav_dir)
            inputs['neg_wav_list'] = wav_list

            list_count: int = len(wav_list)
            inputs['neg_wav_count'] = list_count

            if list_count <= 128:
                inputs['neg_num_thread'] = list_count
            else:
                inputs['neg_num_thread'] = 128

        return inputs
