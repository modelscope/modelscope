# Copyright (c) 2021 Binbin Zhang(binbzha@qq.com)
#               2022 Shaoqing Yu(954793264@qq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os

import json
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torchaudio

from modelscope.utils.logger import get_logger
from .file_utils import make_pair, read_lists

logger = get_logger()

font = fm.FontProperties(size=15)


def load_data_and_score(keywords_list, data_file, trans_file, score_file):
    # score_table: {uttid: [keywordlist]}
    score_table = {}
    with open(score_file, 'r', encoding='utf8') as fin:
        # read score file and store in table
        for line in fin:
            arr = line.strip().split()
            key = arr[0]
            is_detected = arr[1]
            if is_detected == 'detected':
                if key not in score_table:
                    score_table.update(
                        {key: {
                            'kw': arr[2],
                            'confi': float(arr[3])
                        }})
            else:
                if key not in score_table:
                    score_table.update({key: {'kw': 'unknown', 'confi': -1.0}})

    wav_lists = read_lists(data_file)
    trans_lists = read_lists(trans_file)
    data_lists = make_pair(wav_lists, trans_lists)

    # build empty structure for keyword-filler infos
    keyword_filler_table = {}
    for keyword in keywords_list:
        keyword_filler_table[keyword] = {}
        keyword_filler_table[keyword]['keyword_table'] = {}
        keyword_filler_table[keyword]['keyword_duration'] = 0.0
        keyword_filler_table[keyword]['filler_table'] = {}
        keyword_filler_table[keyword]['filler_duration'] = 0.0

    for obj in data_lists:
        assert 'key' in obj
        assert 'wav' in obj
        assert 'txt' in obj
        key = obj['key']
        wav_file = obj['wav']
        txt = obj['txt']
        assert key in score_table

        waveform, rate = torchaudio.load(wav_file)
        frames = len(waveform[0])
        duration = frames / float(rate)

        for keyword in keywords_list:
            if txt.find(keyword) != -1:
                if keyword == score_table[key]['kw']:
                    keyword_filler_table[keyword]['keyword_table'].update(
                        {key: score_table[key]['confi']})
                    keyword_filler_table[keyword][
                        'keyword_duration'] += duration
                else:
                    # uttrance detected but not match this keyword
                    keyword_filler_table[keyword]['keyword_table'].update(
                        {key: -1.0})
                    keyword_filler_table[keyword][
                        'keyword_duration'] += duration
            else:
                keyword_filler_table[keyword]['filler_table'].update(
                    {key: score_table[key]['confi']})
                keyword_filler_table[keyword]['filler_duration'] += duration

    return keyword_filler_table


def load_stats_file(stats_file):
    values = []
    with open(stats_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            threshold, recall, fa_rate, fa_per_hour = arr
            values.append([float(fa_per_hour), (1 - float(recall)) * 100])
    values.reverse()
    return np.array(values)


def compute_det(**kwargs):
    assert kwargs.get('keywords', None) is not None, \
        'Please config param: keywords, preset keyword str, split with \',\''
    keywords = kwargs['keywords']

    assert kwargs.get('test_data', None) is not None, \
        'Please config param: test_data, test waves in list'
    test_data = kwargs['test_data']

    assert kwargs.get('trans_data', None) is not None, \
        'Please config param: trans_data, transcription of test waves'
    trans_data = kwargs['trans_data']

    assert kwargs.get('score_file', None) is not None, \
        'Please config param: score_file, the output scores of test data'
    score_file = kwargs['score_file']

    if kwargs.get('stats_dir', None) is not None:
        stats_dir = kwargs['stats_dir']
    else:
        stats_dir = os.path.dirname(score_file)
    logger.info(f'store all keyword\'s stats file in {stats_dir}')
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)

    score_step = kwargs.get('score_step', 0.001)

    keywords_list = keywords.replace(' ', '').strip().split(',')
    keyword_filler_table = load_data_and_score(keywords_list, test_data,
                                               trans_data, score_file)

    stats_files = {}
    for keyword in keywords_list:
        keyword_dur = keyword_filler_table[keyword]['keyword_duration']
        keyword_num = len(keyword_filler_table[keyword]['keyword_table'])
        filler_dur = keyword_filler_table[keyword]['filler_duration']
        filler_num = len(keyword_filler_table[keyword]['filler_table'])
        assert keyword_num > 0, 'Can\'t compute det for {} without positive sample'
        assert filler_num > 0, 'Can\'t compute det for {} without negative sample'

        logger.info('Computing det for {}'.format(keyword))
        logger.info('  Keyword duration: {} Hours, wave number: {}'.format(
            keyword_dur / 3600.0, keyword_num))
        logger.info('  Filler duration: {} Hours'.format(filler_dur / 3600.0))

        stats_file = os.path.join(stats_dir, 'stats_' + keyword + '.txt')
        with open(stats_file, 'w', encoding='utf8') as fout:
            threshold = 0.0
            while threshold <= 1.0:
                num_false_reject = 0
                num_true_detect = 0
                # transverse the all keyword_table
                for key, confi in keyword_filler_table[keyword][
                        'keyword_table'].items():
                    if confi < threshold:
                        num_false_reject += 1
                    else:
                        num_true_detect += 1

                num_false_alarm = 0
                # transverse the all filler_table
                for key, confi in keyword_filler_table[keyword][
                        'filler_table'].items():
                    if confi >= threshold:
                        num_false_alarm += 1
                        # print(f'false alarm: {keyword}, {key}, {confi}')

                # false_reject_rate = num_false_reject / keyword_num
                true_detect_rate = num_true_detect / keyword_num

                num_false_alarm = max(num_false_alarm, 1e-6)
                false_alarm_per_hour = num_false_alarm / (filler_dur / 3600.0)
                false_alarm_rate = num_false_alarm / filler_num

                fout.write('{:.3f} {:.6f} {:.6f} {:.6f}\n'.format(
                    threshold, true_detect_rate, false_alarm_rate,
                    false_alarm_per_hour))
                threshold += score_step

        stats_files[keyword] = stats_file

    return stats_files


def plot_det(**kwargs):
    assert kwargs.get('dets_dir', None) is not None, \
        'Please config param: dets_dir, to load det files'
    dets_dir = kwargs['dets_dir']

    det_title = kwargs.get('det_title', 'DetCurve')

    assert kwargs.get('figure_file', None) is not None, \
        'Please config param: figure_file, path to save det curve'
    figure_file = kwargs['figure_file']

    xlim = kwargs.get('xlim', '[0,2]')
    # xstep = kwargs.get('xstep', '1')
    ylim = kwargs.get('ylim', '[15,30]')
    # ystep = kwargs.get('ystep', '5')

    plt.figure(dpi=200)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.size'] = 12

    for file in glob.glob(f'{dets_dir}/*stats*.txt'):
        logger.info(f'reading det data from {file}')
        label = os.path.basename(file).split('.')[0]
        values = load_stats_file(file)
        plt.plot(values[:, 0], values[:, 1], label=label)

    xlim_splits = xlim.strip().replace('[', '').replace(']', '').split(',')
    assert len(xlim_splits) == 2
    ylim_splits = ylim.strip().replace('[', '').replace(']', '').split(',')
    assert len(ylim_splits) == 2

    plt.xlim(float(xlim_splits[0]), float(xlim_splits[1]))
    plt.ylim(float(ylim_splits[0]), float(ylim_splits[1]))

    # plt.xticks(range(0, xlim + x_step, x_step))
    # plt.yticks(range(0, ylim + y_step, y_step))
    plt.xlabel('False Alarm Per Hour')
    plt.ylabel('False Rejection Rate (\\%)')
    plt.title(det_title, fontproperties=font)
    plt.grid(linestyle='--')
    # plt.legend(loc='best', fontsize=6)
    plt.legend(loc='upper right', fontsize=5)
    # plt.show()
    plt.savefig(figure_file)
