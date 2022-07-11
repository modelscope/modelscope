import os
from typing import Any, Dict, List

import numpy as np


def type_checking(wav_path: str,
                  recog_type: str = None,
                  audio_format: str = None,
                  workspace: str = None):
    assert os.path.exists(wav_path), f'wav_path:{wav_path} does not exist'

    r_recog_type = recog_type
    r_audio_format = audio_format
    r_workspace = workspace
    r_wav_path = wav_path

    if r_workspace is None or len(r_workspace) == 0:
        r_workspace = os.path.join(os.getcwd(), '.tmp')

    if r_recog_type is None:
        if os.path.isfile(wav_path):
            if wav_path.endswith('.wav') or wav_path.endswith('.WAV'):
                r_recog_type = 'wav'
                r_audio_format = 'wav'

        elif os.path.isdir(wav_path):
            dir_name = os.path.basename(wav_path)
            if 'test' in dir_name:
                r_recog_type = 'test'
            elif 'dev' in dir_name:
                r_recog_type = 'dev'
            elif 'train' in dir_name:
                r_recog_type = 'train'

    if r_audio_format is None:
        if find_file_by_ends(wav_path, '.ark'):
            r_audio_format = 'kaldi_ark'
        elif find_file_by_ends(wav_path, '.wav') or find_file_by_ends(
                wav_path, '.WAV'):
            r_audio_format = 'wav'

    if r_audio_format == 'kaldi_ark' and r_recog_type != 'wav':
        # datasets with kaldi_ark file
        r_wav_path = os.path.abspath(os.path.join(r_wav_path, '../'))
    elif r_audio_format == 'wav' and r_recog_type != 'wav':
        # datasets with waveform files
        r_wav_path = os.path.abspath(os.path.join(r_wav_path, '../../'))

    return r_recog_type, r_audio_format, r_workspace, r_wav_path


def find_file_by_ends(dir_path: str, ends: str):
    dir_files = os.listdir(dir_path)
    for file in dir_files:
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            if file_path.endswith(ends):
                return True
        elif os.path.isdir(file_path):
            if find_file_by_ends(file_path, ends):
                return True

    return False


def compute_wer(hyp_text_path: str, ref_text_path: str) -> Dict[str, Any]:
    assert os.path.exists(hyp_text_path), 'hyp_text does not exist'
    assert os.path.exists(ref_text_path), 'ref_text does not exist'

    rst = {
        'Wrd': 0,
        'Corr': 0,
        'Ins': 0,
        'Del': 0,
        'Sub': 0,
        'Snt': 0,
        'Err': 0.0,
        'S.Err': 0.0,
        'wrong_words': 0,
        'wrong_sentences': 0
    }

    with open(ref_text_path, 'r', encoding='utf-8') as r:
        r_lines = r.readlines()

    with open(hyp_text_path, 'r', encoding='utf-8') as h:
        h_lines = h.readlines()

        for r_line in r_lines:
            r_line_item = r_line.split()
            r_key = r_line_item[0]
            r_sentence = r_line_item[1]
            for h_line in h_lines:
                # find sentence from hyp text
                if r_key in h_line:
                    h_line_item = h_line.split()
                    h_sentence = h_line_item[1]
                    out_item = compute_wer_by_line(h_sentence, r_sentence)
                    rst['Wrd'] += out_item['nwords']
                    rst['Corr'] += out_item['cor']
                    rst['wrong_words'] += out_item['wrong']
                    rst['Ins'] += out_item['ins']
                    rst['Del'] += out_item['del']
                    rst['Sub'] += out_item['sub']
                    rst['Snt'] += 1
                    if out_item['wrong'] > 0:
                        rst['wrong_sentences'] += 1

                    break

        if rst['Wrd'] > 0:
            rst['Err'] = round(rst['wrong_words'] * 100 / rst['Wrd'], 2)
        if rst['Snt'] > 0:
            rst['S.Err'] = round(rst['wrong_sentences'] * 100 / rst['Snt'], 2)

        return rst


def compute_wer_by_line(hyp: list, ref: list) -> Dict[str, Any]:
    len_hyp = len(hyp)
    len_ref = len(ref)
    cost_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int16)

    ops_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int8)

    for i in range(len_hyp + 1):
        cost_matrix[i][0] = i
    for j in range(len_ref + 1):
        cost_matrix[0][j] = j

    for i in range(1, len_hyp + 1):
        for j in range(1, len_ref + 1):
            if hyp[i - 1] == ref[j - 1]:
                cost_matrix[i][j] = cost_matrix[i - 1][j - 1]
            else:
                substitution = cost_matrix[i - 1][j - 1] + 1
                insertion = cost_matrix[i - 1][j] + 1
                deletion = cost_matrix[i][j - 1] + 1

                compare_val = [substitution, insertion, deletion]

                min_val = min(compare_val)
                operation_idx = compare_val.index(min_val) + 1
                cost_matrix[i][j] = min_val
                ops_matrix[i][j] = operation_idx

    match_idx = []
    i = len_hyp
    j = len_ref
    rst = {
        'nwords': len_hyp,
        'cor': 0,
        'wrong': 0,
        'ins': 0,
        'del': 0,
        'sub': 0
    }
    while i >= 0 or j >= 0:
        i_idx = max(0, i)
        j_idx = max(0, j)

        if ops_matrix[i_idx][j_idx] == 0:  # correct
            if i - 1 >= 0 and j - 1 >= 0:
                match_idx.append((j - 1, i - 1))
                rst['cor'] += 1

            i -= 1
            j -= 1

        elif ops_matrix[i_idx][j_idx] == 2:  # insert
            i -= 1
            rst['ins'] += 1

        elif ops_matrix[i_idx][j_idx] == 3:  # delete
            j -= 1
            rst['del'] += 1

        elif ops_matrix[i_idx][j_idx] == 1:  # substitute
            i -= 1
            j -= 1
            rst['sub'] += 1

        if i < 0 and j >= 0:
            rst['del'] += 1
        elif j < 0 and i >= 0:
            rst['ins'] += 1

    match_idx.reverse()
    wrong_cnt = cost_matrix[len_hyp][len_ref]
    rst['wrong'] = wrong_cnt

    return rst
