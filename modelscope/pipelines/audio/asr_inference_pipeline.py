# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, List, Sequence, Tuple, Union

import yaml

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import WavToScp
from modelscope.utils.audio.audio_utils import (extract_pcm_from_wav,
                                                generate_scp_from_url,
                                                load_bytes_from_url)
from modelscope.utils.constant import Frameworks, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['AutomaticSpeechRecognitionPipeline']


@PIPELINES.register_module(
    Tasks.auto_speech_recognition, module_name=Pipelines.asr_inference)
class AutomaticSpeechRecognitionPipeline(Pipeline):
    """ASR Inference Pipeline
    Example:

    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.utils.constant import Tasks

    >>> inference_pipeline = pipeline(
    >>>     task=Tasks.auto_speech_recognition,
    >>>     model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')

    >>> rec_result = inference_pipeline(
    >>>     audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav')
    >>> print(rec_result)

    """

    def __init__(self,
                 model: Union[Model, str] = None,
                 preprocessor: WavToScp = None,
                 **kwargs):
        """
        Use `model` and `preprocessor` to create an asr pipeline for prediction
        Args:
            model ('Model' or 'str'):
                The pipeline handles three types of model:

                - A model instance
                - A model local dir
                - A model id in the model hub
            preprocessor:
                (list of) Preprocessor object
            output_dir('str'):
                output dir path
            batch_size('int'):
                the batch size for inference
            ngpu('int'):
                the number of gpus, 0 indicates CPU mode
            beam_size('int'):
                beam size for decoding
            ctc_weight('float'):
                CTC weight in joint decoding
            lm_weight('float'):
                lm weight
            decoding_ind('int', defaults to 0):
                decoding ind
            decoding_mode('str', defaults to 'model1'):
                decoding mode
            vad_model_file('str'):
                vad model file
            vad_infer_config('str'):
                VAD infer configuration
            vad_cmvn_file('str'):
                global CMVN file
            punc_model_file('str'):
                punc model file
            punc_infer_config('str'):
                punc infer config
            param_dict('dict'):
                extra kwargs
        """
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.model_cfg = self.model.forward()

        self.cmd = self.get_cmd(kwargs)
        if self.cmd['code_base'] == 'funasr':
            from funasr.bin import asr_inference_launch
            self.funasr_infer_modelscope = asr_inference_launch.inference_launch(
                mode=self.cmd['mode'],
                maxlenratio=self.cmd['maxlenratio'],
                minlenratio=self.cmd['minlenratio'],
                batch_size=self.cmd['batch_size'],
                beam_size=self.cmd['beam_size'],
                ngpu=self.cmd['ngpu'],
                ctc_weight=self.cmd['ctc_weight'],
                lm_weight=self.cmd['lm_weight'],
                penalty=self.cmd['penalty'],
                log_level=self.cmd['log_level'],
                asr_train_config=self.cmd['asr_train_config'],
                asr_model_file=self.cmd['asr_model_file'],
                cmvn_file=self.cmd['cmvn_file'],
                lm_file=self.cmd['lm_file'],
                token_type=self.cmd['token_type'],
                key_file=self.cmd['key_file'],
                word_lm_train_config=self.cmd['word_lm_train_config'],
                bpemodel=self.cmd['bpemodel'],
                allow_variable_data_keys=self.cmd['allow_variable_data_keys'],
                output_dir=self.cmd['output_dir'],
                dtype=self.cmd['dtype'],
                seed=self.cmd['seed'],
                ngram_weight=self.cmd['ngram_weight'],
                nbest=self.cmd['nbest'],
                num_workers=self.cmd['num_workers'],
                vad_infer_config=self.cmd['vad_infer_config'],
                vad_model_file=self.cmd['vad_model_file'],
                vad_cmvn_file=self.cmd['vad_cmvn_file'],
                punc_model_file=self.cmd['punc_model_file'],
                punc_infer_config=self.cmd['punc_infer_config'],
                outputs_dict=self.cmd['outputs_dict'],
                param_dict=self.cmd['param_dict'],
                token_num_relax=self.cmd['token_num_relax'],
                decoding_ind=self.cmd['decoding_ind'],
                decoding_mode=self.cmd['decoding_mode'],
            )

    def __call__(self,
                 audio_in: Union[str, bytes],
                 audio_fs: int = None,
                 recog_type: str = None,
                 audio_format: str = None,
                 output_dir: str = None,
                 param_dict: dict = None) -> Dict[str, Any]:
        from funasr.utils import asr_utils
        """
        Decoding the input audios
        Args:
            audio_in('str' or 'bytes'):
                - A string containing a local path to a wav file
                - A string containing a local path to a scp
                - A string containing a wav url
                - A bytes input
            audio_fs('int'):
                frequency of sample
            recog_type('str'):
                recog type
            audio_format('str'):
                audio format
            output_dir('str'):
                output dir
            param_dict('dict'):
                extra kwargs
        Return:
            A dictionary of result or a list of dictionary of result.

            The dictionary contain the following keys:
            - **text** ('str') --The asr result.
        """

        # code base
        code_base = self.cmd['code_base']
        self.recog_type = recog_type
        self.audio_format = audio_format
        self.audio_fs = audio_fs
        checking_audio_fs = None
        self.raw_inputs = None
        if output_dir is not None:
            self.cmd['output_dir'] = output_dir
        if audio_fs is not None:
            self.cmd['fs']['audio_fs'] = audio_fs
        self.cmd['param_dict'] = param_dict

        if code_base == 'funasr':
            if isinstance(audio_in, str):
                # for funasr code, generate wav.scp from url or local path
                self.audio_in, self.raw_inputs = generate_scp_from_url(
                    audio_in)
            elif isinstance(audio_in, bytes):
                self.audio_in = audio_in
                self.raw_inputs = None
            else:
                import numpy
                import torch
                if isinstance(audio_in, torch.Tensor):
                    self.audio_in = None
                    self.raw_inputs = audio_in
                elif isinstance(audio_in, numpy.ndarray):
                    self.audio_in = None
                    self.raw_inputs = audio_in
        elif isinstance(audio_in, str):
            # load pcm data from url if audio_in is url str
            self.audio_in, checking_audio_fs = load_bytes_from_url(audio_in)
        elif isinstance(audio_in, bytes):
            # load pcm data from wav data if audio_in is wave format
            self.audio_in, checking_audio_fs = extract_pcm_from_wav(audio_in)
        else:
            self.audio_in = audio_in

        # set the sample_rate of audio_in if checking_audio_fs is valid
        if checking_audio_fs is not None:
            self.audio_fs = checking_audio_fs

        if recog_type is None or audio_format is None:
            self.recog_type, self.audio_format, self.audio_in = asr_utils.type_checking(
                audio_in=self.audio_in,
                recog_type=recog_type,
                audio_format=audio_format)

        if hasattr(asr_utils,
                   'sample_rate_checking') and self.audio_in is not None:
            checking_audio_fs = asr_utils.sample_rate_checking(
                self.audio_in, self.audio_format)
            if checking_audio_fs is not None:
                self.audio_fs = checking_audio_fs

        output = self.preprocessor.forward(self.model_cfg, self.recog_type,
                                           self.audio_format, self.audio_in,
                                           self.audio_fs, self.cmd)
        output = self.forward(output)
        rst = self.postprocess(output)
        return rst

    def get_cmd(self, extra_args) -> Dict[str, Any]:
        if self.preprocessor is None:
            self.preprocessor = WavToScp()

        outputs = self.preprocessor.config_checking(self.model_cfg)
        # generate asr inference command
        cmd = {
            'maxlenratio': 0.0,
            'minlenratio': 0.0,
            'batch_size': 1,
            'beam_size': 1,
            'ngpu': 1,
            'ctc_weight': 0.0,
            'lm_weight': 0.0,
            'penalty': 0.0,
            'log_level': 'ERROR',
            'asr_train_config': None,
            'asr_model_file': outputs['am_model_path'],
            'cmvn_file': None,
            'lm_train_config': None,
            'lm_file': None,
            'token_type': None,
            'key_file': None,
            'word_lm_train_config': None,
            'bpemodel': None,
            'allow_variable_data_keys': False,
            'output_dir': None,
            'dtype': 'float32',
            'seed': 0,
            'ngram_weight': 0.9,
            'nbest': 1,
            'num_workers': 1,
            'vad_infer_config': None,
            'vad_model_file': None,
            'vad_cmvn_file': None,
            'time_stamp_writer': True,
            'punc_infer_config': None,
            'punc_model_file': None,
            'outputs_dict': True,
            'param_dict': None,
            'model_type': outputs['model_type'],
            'idx_text': '',
            'sampled_ids': 'seq2seq/sampled_ids',
            'sampled_lengths': 'seq2seq/sampled_lengths',
            'lang': 'zh-cn',
            'code_base': outputs['code_base'],
            'mode': outputs['mode'],
            'fs': {
                'model_fs': None,
                'audio_fs': None
            }
        }

        if self.framework == Frameworks.torch:
            frontend_conf = None
            token_num_relax = None
            decoding_ind = None
            decoding_mode = None
            if os.path.exists(outputs['am_model_config']):
                config_file = open(
                    outputs['am_model_config'], encoding='utf-8')
                root = yaml.full_load(config_file)
                config_file.close()
                if 'frontend_conf' in root:
                    frontend_conf = root['frontend_conf']
            if os.path.exists(outputs['asr_model_config']):
                config_file = open(
                    outputs['asr_model_config'], encoding='utf-8')
                root = yaml.full_load(config_file)
                config_file.close()
                if 'token_num_relax' in root:
                    token_num_relax = root['token_num_relax']
                if 'decoding_ind' in root:
                    decoding_ind = root['decoding_ind']
                if 'decoding_mode' in root:
                    decoding_mode = root['decoding_mode']

                cmd['beam_size'] = root['beam_size']
                cmd['penalty'] = root['penalty']
                cmd['maxlenratio'] = root['maxlenratio']
                cmd['minlenratio'] = root['minlenratio']
                cmd['ctc_weight'] = root['ctc_weight']
                cmd['lm_weight'] = root['lm_weight']
            else:
                # for vad task, no asr_model_config
                cmd['beam_size'] = None
                cmd['penalty'] = None
                cmd['maxlenratio'] = None
                cmd['minlenratio'] = None
                cmd['ctc_weight'] = None
                cmd['lm_weight'] = None
            cmd['asr_train_config'] = outputs['am_model_config']
            cmd['lm_file'] = outputs['lm_model_path']
            cmd['lm_train_config'] = outputs['lm_model_config']
            cmd['batch_size'] = outputs['model_config']['batch_size']
            cmd['frontend_conf'] = frontend_conf
            if frontend_conf is not None and 'fs' in frontend_conf:
                cmd['fs']['model_fs'] = frontend_conf['fs']
            cmd['token_num_relax'] = token_num_relax
            cmd['decoding_ind'] = decoding_ind
            cmd['decoding_mode'] = decoding_mode
            if outputs.__contains__('mvn_file'):
                cmd['cmvn_file'] = outputs['mvn_file']
            if outputs.__contains__('vad_model_name'):
                cmd['vad_model_file'] = outputs['vad_model_name']
            if outputs.__contains__('vad_model_config'):
                cmd['vad_infer_config'] = outputs['vad_model_config']
            if outputs.__contains__('vad_mvn_file'):
                cmd['vad_cmvn_file'] = outputs['vad_mvn_file']
            if outputs.__contains__('punc_model_name'):
                cmd['punc_model_file'] = outputs['punc_model_name']
            if outputs.__contains__('punc_model_config'):
                cmd['punc_infer_config'] = outputs['punc_model_config']

            user_args_dict = [
                'output_dir',
                'batch_size',
                'mode',
                'ngpu',
                'beam_size',
                'ctc_weight',
                'lm_weight',
                'decoding_ind',
                'decoding_mode',
                'vad_model_file',
                'vad_infer_config',
                'vad_cmvn_file',
                'punc_model_file',
                'punc_infer_config',
                'param_dict',
            ]

            for user_args in user_args_dict:
                if user_args in extra_args and extra_args[
                        user_args] is not None:
                    cmd[user_args] = extra_args[user_args]

        elif self.framework == Frameworks.tf:
            cmd['fs']['model_fs'] = outputs['model_config']['fs']
            cmd['hop_length'] = outputs['model_config']['hop_length']
            cmd['feature_dims'] = outputs['model_config']['feature_dims']
            cmd['predictions_file'] = 'text'
            cmd['cmvn_file'] = outputs['am_mvn_file']
            cmd['vocab_file'] = outputs['vocab_file']
            if 'idx_text' in outputs:
                cmd['idx_text'] = outputs['idx_text']
            if 'sampled_ids' in outputs['model_config']:
                cmd['sampled_ids'] = outputs['model_config']['sampled_ids']
            if 'sampled_lengths' in outputs['model_config']:
                cmd['sampled_lengths'] = outputs['model_config'][
                    'sampled_lengths']
        else:
            raise ValueError('model type is mismatching')

        return cmd

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Decoding
        """

        logger.info(f"Decoding with {inputs['audio_format']} files ...")

        data_cmd: Sequence[Tuple[str, str, str]]
        if self.cmd['code_base'] == 'funasr':
            if isinstance(self.audio_in, bytes):
                data_cmd = [self.audio_in, 'speech', 'bytes']
            elif isinstance(self.audio_in, str):
                data_cmd = [self.audio_in, 'speech', 'sound']
            elif self.raw_inputs is not None:
                data_cmd = None
        else:
            if inputs['audio_format'] == 'wav' or inputs[
                    'audio_format'] == 'pcm':
                data_cmd = ['speech', 'sound']
            elif inputs['audio_format'] == 'kaldi_ark':
                data_cmd = ['speech', 'kaldi_ark']
            elif inputs['audio_format'] == 'tfrecord':
                data_cmd = ['speech', 'tfrecord']
            if inputs.__contains__('mvn_file'):
                data_cmd.append(inputs['mvn_file'])

        # generate asr inference command
        self.cmd['name_and_type'] = data_cmd
        self.cmd['raw_inputs'] = self.raw_inputs
        self.cmd['audio_in'] = self.audio_in

        inputs['asr_result'] = self.run_inference(self.cmd)

        return inputs

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """process the asr results
        """
        from funasr.utils import asr_utils

        logger.info('Computing the result of ASR ...')

        rst = {}

        # single wav or pcm task
        if inputs['recog_type'] == 'wav':
            if 'asr_result' in inputs and len(inputs['asr_result']) > 0:
                for key, value in inputs['asr_result'][0].items():
                    if key == 'value':
                        if len(value) > 0:
                            rst[OutputKeys.TEXT] = value
                    elif key != 'key':
                        rst[key] = value

        # run with datasets, and audio format is waveform or kaldi_ark or tfrecord
        elif inputs['recog_type'] != 'wav':
            inputs['reference_list'] = self.ref_list_tidy(inputs)

            inputs['datasets_result'] = asr_utils.compute_wer(
                hyp_list=inputs['asr_result'],
                ref_list=inputs['reference_list'])

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
                line_item = line.split(None, 1)
                if len(line_item) > 1:
                    item = {
                        'key': line_item[0],
                        'value': line_item[1].strip('\n')
                    }
                    ref_list.append(item)

        return ref_list

    def run_inference(self, cmd):
        asr_result = []
        if self.framework == Frameworks.torch and cmd['code_base'] == 'funasr':
            asr_result = self.funasr_infer_modelscope(
                data_path_and_name_and_type=cmd['name_and_type'],
                raw_inputs=cmd['raw_inputs'],
                output_dir_v2=cmd['output_dir'],
                fs=cmd['fs'],
                param_dict=cmd['param_dict'])

        elif self.framework == Frameworks.tf:
            from easyasr import asr_inference_paraformer_tf
            if hasattr(asr_inference_paraformer_tf, 'set_parameters'):
                asr_inference_paraformer_tf.set_parameters(
                    language=cmd['lang'])
            else:
                # in order to support easyasr-0.0.2
                cmd['fs'] = cmd['fs']['model_fs']

            asr_result = asr_inference_paraformer_tf.asr_inference(
                ngpu=cmd['ngpu'],
                name_and_type=cmd['name_and_type'],
                audio_lists=cmd['audio_in'],
                idx_text_file=cmd['idx_text'],
                asr_model_file=cmd['asr_model_file'],
                vocab_file=cmd['vocab_file'],
                am_mvn_file=cmd['cmvn_file'],
                predictions_file=cmd['predictions_file'],
                fs=cmd['fs'],
                hop_length=cmd['hop_length'],
                feature_dims=cmd['feature_dims'],
                sampled_ids=cmd['sampled_ids'],
                sampled_lengths=cmd['sampled_lengths'])

        return asr_result
