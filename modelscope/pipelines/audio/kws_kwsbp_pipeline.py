import io
import os
import shutil
import stat
import subprocess
from typing import Any, Dict, List, Union

import json

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import WavToLists
from modelscope.utils.constant import Tasks

__all__ = ['KeyWordSpottingKwsbpPipeline']


@PIPELINES.register_module(
    Tasks.key_word_spotting, module_name=Pipelines.kws_kwsbp)
class KeyWordSpottingKwsbpPipeline(Pipeline):
    """KWS Pipeline - key word spotting decoding
    """

    def __init__(self,
                 config_file: str = None,
                 model: Union[Model, str] = None,
                 preprocessor: WavToLists = None,
                 **kwargs):
        """use `model` and `preprocessor` to create a kws pipeline for prediction
        """

        model = model if isinstance(model,
                                    Model) else Model.from_pretrained(model)

        super().__init__(
            config_file=config_file,
            model=model,
            preprocessor=preprocessor,
            **kwargs)
        assert model is not None, 'kws model should be provided'

        self._preprocessor = preprocessor
        self._model = model
        self._keywords = None

        if 'keywords' in kwargs.keys():
            self._keywords = kwargs['keywords']

    def __call__(self,
                 kws_type: str,
                 wav_path: List[str],
                 workspace: str = None) -> Dict[str, Any]:
        assert kws_type in ['wav', 'pos_testsets', 'neg_testsets',
                            'roc'], f'kws_type {kws_type} is invalid'

        if self._preprocessor is None:
            self._preprocessor = WavToLists(workspace=workspace)

        output = self._preprocessor.forward(self._model.forward(), kws_type,
                                            wav_path)
        output = self.forward(output)
        rst = self.postprocess(output)
        return rst

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Decoding
        """

        # will generate kws result into dump/dump.JOB.log
        out = self._run_with_kwsbp(inputs)

        return out

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """process the kws results
        """

        pos_result_json = {}
        neg_result_json = {}

        if inputs['kws_set'] in ['wav', 'pos_testsets', 'roc']:
            self._parse_dump_log(pos_result_json, inputs['pos_dump_path'])
        if inputs['kws_set'] in ['neg_testsets', 'roc']:
            self._parse_dump_log(neg_result_json, inputs['neg_dump_path'])
        """
        result_json format example:
            {
                "wav_count": 450,
                "keywords": ["小云小云"],
                "wav_time": 3560.999999,
                "detected": [
                    {
                        "xxx.wav": {
                            "confidence": "0.990368",
                            "keyword": "小云小云"
                        }
                    },
                    {
                        "yyy.wav": {
                            "confidence": "0.990368",
                            "keyword": "小云小云"
                        }
                    },
                    ......
                ],
                "detected_count": 429,
                "rejected_count": 21,
                "rejected": [
                    "yyy.wav",
                    "zzz.wav",
                    ......
                ]
            }
        """

        rst_dict = {'kws_set': inputs['kws_set']}

        # parsing the result of wav
        if inputs['kws_set'] == 'wav':
            rst_dict['wav_count'] = pos_result_json['wav_count'] = inputs[
                'pos_wav_count']
            rst_dict['wav_time'] = round(pos_result_json['wav_time'], 6)
            if pos_result_json['detected_count'] == 1:
                rst_dict['keywords'] = pos_result_json['keywords']
                rst_dict['detected'] = True
                wav_file_name = os.path.basename(inputs['pos_wav_path'])
                rst_dict['confidence'] = float(pos_result_json['detected'][0]
                                               [wav_file_name]['confidence'])
            else:
                rst_dict['detected'] = False

        # parsing the result of pos_tests
        elif inputs['kws_set'] == 'pos_testsets':
            rst_dict['wav_count'] = pos_result_json['wav_count'] = inputs[
                'pos_wav_count']
            rst_dict['wav_time'] = round(pos_result_json['wav_time'], 6)
            if pos_result_json.__contains__('keywords'):
                rst_dict['keywords'] = pos_result_json['keywords']

            rst_dict['recall'] = round(
                pos_result_json['detected_count'] / rst_dict['wav_count'], 6)

            if pos_result_json.__contains__('detected_count'):
                rst_dict['detected_count'] = pos_result_json['detected_count']
            if pos_result_json.__contains__('rejected_count'):
                rst_dict['rejected_count'] = pos_result_json['rejected_count']
            if pos_result_json.__contains__('rejected'):
                rst_dict['rejected'] = pos_result_json['rejected']

        # parsing the result of neg_tests
        elif inputs['kws_set'] == 'neg_testsets':
            rst_dict['wav_count'] = neg_result_json['wav_count'] = inputs[
                'neg_wav_count']
            rst_dict['wav_time'] = round(neg_result_json['wav_time'], 6)
            if neg_result_json.__contains__('keywords'):
                rst_dict['keywords'] = neg_result_json['keywords']

            rst_dict['fa_rate'] = 0.0
            rst_dict['fa_per_hour'] = 0.0

            if neg_result_json.__contains__('detected_count'):
                rst_dict['detected_count'] = neg_result_json['detected_count']
                rst_dict['fa_rate'] = round(
                    neg_result_json['detected_count'] / rst_dict['wav_count'],
                    6)
                if neg_result_json.__contains__('wav_time'):
                    rst_dict['fa_per_hour'] = round(
                        neg_result_json['detected_count']
                        / float(neg_result_json['wav_time'] / 3600), 6)

            if neg_result_json.__contains__('rejected_count'):
                rst_dict['rejected_count'] = neg_result_json['rejected_count']

            if neg_result_json.__contains__('detected'):
                rst_dict['detected'] = neg_result_json['detected']

        # parsing the result of roc
        elif inputs['kws_set'] == 'roc':
            threshold_start = 0.000
            threshold_step = 0.001
            threshold_end = 1.000

            pos_keywords_list = []
            neg_keywords_list = []
            if pos_result_json.__contains__('keywords'):
                pos_keywords_list = pos_result_json['keywords']
            if neg_result_json.__contains__('keywords'):
                neg_keywords_list = neg_result_json['keywords']

            keywords_list = list(set(pos_keywords_list + neg_keywords_list))

            pos_result_json['wav_count'] = inputs['pos_wav_count']
            neg_result_json['wav_count'] = inputs['neg_wav_count']

            if len(keywords_list) > 0:
                rst_dict['keywords'] = keywords_list

            for index in range(len(rst_dict['keywords'])):
                cur_keyword = rst_dict['keywords'][index]
                output_list = self._generate_roc_list(
                    start=threshold_start,
                    step=threshold_step,
                    end=threshold_end,
                    keyword=cur_keyword,
                    pos_inputs=pos_result_json,
                    neg_inputs=neg_result_json)

                rst_dict[cur_keyword] = output_list

        return rst_dict

    def _run_with_kwsbp(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        opts: str = ''

        # setting customized keywords
        keywords_json = self._set_customized_keywords()
        if len(keywords_json) > 0:
            keywords_json_file = os.path.join(inputs['workspace'],
                                              'keyword_custom.json')
            with open(keywords_json_file, 'w') as f:
                json.dump(keywords_json, f)
            opts = '--keyword-custom ' + keywords_json_file

        if inputs['kws_set'] == 'roc':
            inputs['keyword_grammar_path'] = os.path.join(
                inputs['model_workspace'], 'keywords_roc.json')

        if inputs['kws_set'] == 'wav':
            dump_log_path: str = os.path.join(inputs['pos_dump_path'],
                                              'dump.log')
            kws_cmd: str = inputs['kws_tool_path'] + \
                ' --sys-dir=' + inputs['model_workspace'] + \
                ' --cfg-file=' + inputs['cfg_file_path'] + \
                ' --sample-rate=' + inputs['sample_rate'] + \
                ' --keyword-grammar=' + inputs['keyword_grammar_path'] + \
                ' --wave-scp=' + os.path.join(inputs['pos_data_path'], 'wave.list') + \
                ' --num-thread=1 ' + opts + ' > ' + dump_log_path + ' 2>&1'
            os.system(kws_cmd)

        if inputs['kws_set'] in ['pos_testsets', 'roc']:
            data_dir: str = os.listdir(inputs['pos_data_path'])
            wav_list = []
            for i in data_dir:
                suffix = os.path.splitext(os.path.basename(i))[1]
                if suffix == '.list':
                    wav_list.append(os.path.join(inputs['pos_data_path'], i))

            j: int = 0
            process = []
            while j < inputs['pos_num_thread']:
                wav_list_path: str = inputs['pos_data_path'] + '/wave.' + str(
                    j) + '.list'
                dump_log_path: str = inputs['pos_dump_path'] + '/dump.' + str(
                    j) + '.log'

                kws_cmd: str = inputs['kws_tool_path'] + \
                    ' --sys-dir=' + inputs['model_workspace'] + \
                    ' --cfg-file=' + inputs['cfg_file_path'] + \
                    ' --sample-rate=' + inputs['sample_rate'] + \
                    ' --keyword-grammar=' + inputs['keyword_grammar_path'] + \
                    ' --wave-scp=' + wav_list_path + \
                    ' --num-thread=1 ' + opts + ' > ' + dump_log_path + ' 2>&1'
                p = subprocess.Popen(kws_cmd, shell=True)
                process.append(p)
                j += 1

            k: int = 0
            while k < len(process):
                process[k].wait()
                k += 1

        if inputs['kws_set'] in ['neg_testsets', 'roc']:
            data_dir: str = os.listdir(inputs['neg_data_path'])
            wav_list = []
            for i in data_dir:
                suffix = os.path.splitext(os.path.basename(i))[1]
                if suffix == '.list':
                    wav_list.append(os.path.join(inputs['neg_data_path'], i))

            j: int = 0
            process = []
            while j < inputs['neg_num_thread']:
                wav_list_path: str = inputs['neg_data_path'] + '/wave.' + str(
                    j) + '.list'
                dump_log_path: str = inputs['neg_dump_path'] + '/dump.' + str(
                    j) + '.log'

                kws_cmd: str = inputs['kws_tool_path'] + \
                    ' --sys-dir=' + inputs['model_workspace'] + \
                    ' --cfg-file=' + inputs['cfg_file_path'] + \
                    ' --sample-rate=' + inputs['sample_rate'] + \
                    ' --keyword-grammar=' + inputs['keyword_grammar_path'] + \
                    ' --wave-scp=' + wav_list_path + \
                    ' --num-thread=1 ' + opts + ' > ' + dump_log_path + ' 2>&1'
                p = subprocess.Popen(kws_cmd, shell=True)
                process.append(p)
                j += 1

            k: int = 0
            while k < len(process):
                process[k].wait()
                k += 1

        return inputs

    def _parse_dump_log(self, result_json: Dict[str, Any],
                        dump_path: str) -> Dict[str, Any]:
        dump_dir = os.listdir(dump_path)
        for i in dump_dir:
            basename = os.path.splitext(os.path.basename(i))[0]
            # find dump.JOB.log
            if 'dump' in basename:
                with open(
                        os.path.join(dump_path, i), mode='r',
                        encoding='utf-8') as file:
                    while 1:
                        line = file.readline()
                        if not line:
                            break
                        else:
                            result_json = self._parse_result_log(
                                line, result_json)

    def _parse_result_log(self, line: str,
                          result_json: Dict[str, Any]) -> Dict[str, Any]:
        # valid info
        if '[rejected]' in line or '[detected]' in line:
            detected_count = 0
            rejected_count = 0

            if result_json.__contains__('detected_count'):
                detected_count = result_json['detected_count']
            if result_json.__contains__('rejected_count'):
                rejected_count = result_json['rejected_count']

            if '[detected]' in line:
                # [detected], fname:/xxx/.tmp_pos_testsets/pos_testsets/33.wav,
                # kw:小云小云, confidence:0.965155, time:[4.62-5.10], threshold:0.00,
                detected_count += 1
                content_list = line.split(', ')
                file_name = os.path.basename(content_list[1].split(':')[1])
                keyword = content_list[2].split(':')[1]
                confidence = content_list[3].split(':')[1]

                keywords_list = []
                if result_json.__contains__('keywords'):
                    keywords_list = result_json['keywords']

                if keyword not in keywords_list:
                    keywords_list.append(keyword)
                result_json['keywords'] = keywords_list

                keyword_item = {}
                keyword_item['confidence'] = confidence
                keyword_item['keyword'] = keyword
                item = {}
                item[file_name] = keyword_item

                detected_list = []
                if result_json.__contains__('detected'):
                    detected_list = result_json['detected']

                detected_list.append(item)
                result_json['detected'] = detected_list

            elif '[rejected]' in line:
                # [rejected], fname:/xxx/.tmp_pos_testsets/pos_testsets/28.wav
                rejected_count += 1
                content_list = line.split(', ')
                file_name = os.path.basename(content_list[1].split(':')[1])
                file_name = file_name.strip().replace('\n',
                                                      '').replace('\r', '')

                rejected_list = []
                if result_json.__contains__('rejected'):
                    rejected_list = result_json['rejected']

                rejected_list.append(file_name)
                result_json['rejected'] = rejected_list

            result_json['detected_count'] = detected_count
            result_json['rejected_count'] = rejected_count

        elif 'total_proc_time=' in line and 'wav_time=' in line:
            # eg: total_proc_time=0.289000(s), wav_time=20.944125(s), kwsbp_rtf=0.013799
            wav_total_time = 0
            content_list = line.split('), ')
            if result_json.__contains__('wav_time'):
                wav_total_time = result_json['wav_time']

            wav_time_str = content_list[1].split('=')[1]
            wav_time_str = wav_time_str.split('(')[0]
            wav_time = float(wav_time_str)
            wav_time = round(wav_time, 6)

            if isinstance(wav_time, float):
                wav_total_time += wav_time

            result_json['wav_time'] = wav_total_time

        return result_json

    def _generate_roc_list(self, start: float, step: float, end: float,
                           keyword: str, pos_inputs: Dict[str, Any],
                           neg_inputs: Dict[str, Any]) -> Dict[str, Any]:
        pos_wav_count = pos_inputs['wav_count']
        neg_wav_time = neg_inputs['wav_time']
        det_lists = pos_inputs['detected']
        fa_lists = neg_inputs['detected']
        threshold_cur = start
        """
        input det_lists dict
                [
                    {
                        "xxx.wav": {
                            "confidence": "0.990368",
                            "keyword": "小云小云"
                        }
                    },
                    {
                        "yyy.wav": {
                            "confidence": "0.990368",
                            "keyword": "小云小云"
                        }
                    },
                ]

        output dict
                [
                    {
                        "threshold": 0.000,
                        "recall": 0.999888,
                        "fa_per_hour": 1.999999
                    },
                    {
                        "threshold": 0.001,
                        "recall": 0.999888,
                        "fa_per_hour": 1.999999
                    },
                ]
        """

        output = []
        while threshold_cur <= end:
            det_count = 0
            fa_count = 0
            for index in range(len(det_lists)):
                det_item = det_lists[index]
                det_wav_item = det_item.get(next(iter(det_item)))
                if det_wav_item['keyword'] == keyword:
                    confidence = float(det_wav_item['confidence'])
                    if confidence >= threshold_cur:
                        det_count += 1

            for index in range(len(fa_lists)):
                fa_item = fa_lists[index]
                fa_wav_item = fa_item.get(next(iter(fa_item)))
                if fa_wav_item['keyword'] == keyword:
                    confidence = float(fa_wav_item['confidence'])
                    if confidence >= threshold_cur:
                        fa_count += 1

            output_item = {
                'threshold': round(threshold_cur, 3),
                'recall': round(float(det_count / pos_wav_count), 6),
                'fa_per_hour': round(fa_count / float(neg_wav_time / 3600), 6)
            }
            output.append(output_item)

            threshold_cur += step

        return output

    def _set_customized_keywords(self) -> Dict[str, Any]:
        if self._keywords is not None:
            word_list_inputs = self._keywords
            word_list = []
            for i in range(len(word_list_inputs)):
                key = word_list_inputs[i]
                new_item = {}
                if key.__contains__('keyword'):
                    name = key['keyword']
                    new_name: str = ''
                    for n in range(0, len(name), 1):
                        new_name += name[n]
                        new_name += ' '
                    new_name = new_name.strip()
                    new_item['name'] = new_name

                    if key.__contains__('threshold'):
                        threshold1: float = key['threshold']
                        new_item['threshold1'] = threshold1

                word_list.append(new_item)
            out = {'word_list': word_list}
            return out
        else:
            return ''
