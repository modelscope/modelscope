import os
from typing import Any, Dict, List, Union

import json

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import WavToLists
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['KeyWordSpottingKwsbpPipeline']


@PIPELINES.register_module(
    Tasks.auto_speech_recognition, module_name=Pipelines.kws_kwsbp)
class KeyWordSpottingKwsbpPipeline(Pipeline):
    """KWS Pipeline - key word spotting decoding
    """

    def __init__(self,
                 model: Union[Model, str] = None,
                 preprocessor: WavToLists = None,
                 **kwargs):
        """use `model` and `preprocessor` to create a kws pipeline for prediction
        """
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)

    def __call__(self, wav_path: Union[List[str], str],
                 **kwargs) -> Dict[str, Any]:
        if 'keywords' in kwargs.keys():
            self.keywords = kwargs['keywords']
        else:
            self.keywords = None

        if self.preprocessor is None:
            self.preprocessor = WavToLists()

        output = self.preprocessor.forward(self.model.forward(), wav_path)
        output = self.forward(output)
        rst = self.postprocess(output)
        return rst

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Decoding
        """

        logger.info(f"Decoding with {inputs['kws_set']} mode ...")

        # will generate kws result
        out = self.run_with_kwsbp(inputs)

        return out

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """process the kws results

        Args:
          inputs['pos_kws_list'] or inputs['neg_kws_list']:
          result_dict format example:
            [{
              'confidence': 0.9903678297996521,
              'filename': 'data/test/audios/kws_xiaoyunxiaoyun.wav',
              'keyword': '小云小云',
              'offset': 5.760000228881836,  # second
              'rtf_time': 66,               # millisecond
              'threshold': 0,
              'wav_time': 9.1329375         # second
            }]
        """

        import kws_util.common
        neg_kws_list = None
        pos_kws_list = None
        if 'pos_kws_list' in inputs:
            pos_kws_list = inputs['pos_kws_list']
        if 'neg_kws_list' in inputs:
            neg_kws_list = inputs['neg_kws_list']
        rst_dict = kws_util.common.parsing_kws_result(
            kws_type=inputs['kws_set'],
            pos_list=pos_kws_list,
            neg_list=neg_kws_list)

        return rst_dict

    def run_with_kwsbp(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        cmd = {
            'sys_dir': inputs['model_workspace'],
            'cfg_file': inputs['cfg_file_path'],
            'sample_rate': inputs['sample_rate'],
            'keyword_custom': ''
        }

        import kwsbp
        import kws_util.common
        kws_inference = kwsbp.KwsbpEngine()

        # setting customized keywords
        cmd['customized_keywords'] = kws_util.common.generate_customized_keywords(
            self.keywords)

        if inputs['kws_set'] == 'roc':
            inputs['keyword_grammar_path'] = os.path.join(
                inputs['model_workspace'], 'keywords_roc.json')

        if inputs['kws_set'] in ['wav', 'pos_testsets', 'roc']:
            cmd['wave_scp'] = inputs['pos_wav_list']
            cmd['keyword_grammar_path'] = inputs['keyword_grammar_path']
            cmd['num_thread'] = inputs['pos_num_thread']

            # run and get inference result
            result = kws_inference.inference(cmd['sys_dir'], cmd['cfg_file'],
                                             cmd['keyword_grammar_path'],
                                             str(json.dumps(cmd['wave_scp'])),
                                             str(cmd['customized_keywords']),
                                             cmd['sample_rate'],
                                             cmd['num_thread'])
            pos_result = json.loads(result)
            inputs['pos_kws_list'] = pos_result['kws_list']

        if inputs['kws_set'] in ['neg_testsets', 'roc']:
            cmd['wave_scp'] = inputs['neg_wav_list']
            cmd['keyword_grammar_path'] = inputs['keyword_grammar_path']
            cmd['num_thread'] = inputs['neg_num_thread']

            # run and get inference result
            result = kws_inference.inference(cmd['sys_dir'], cmd['cfg_file'],
                                             cmd['keyword_grammar_path'],
                                             str(json.dumps(cmd['wave_scp'])),
                                             str(cmd['customized_keywords']),
                                             cmd['sample_rate'],
                                             cmd['num_thread'])
            neg_result = json.loads(result)
            inputs['neg_kws_list'] = neg_result['kws_list']

        return inputs
