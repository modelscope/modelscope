# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import random

from modelscope.utils.logger import get_logger

logging = get_logger()


def is_fp_line(line):
    fp_category_list = ['FP', 'I', 'N', 'Q']
    elements = line.strip().split(' ')
    res = True
    for ele in elements:
        if ele not in fp_category_list:
            res = False
            break
    return res


class FpProcessor:

    def __init__(self):
        #  TODO: Add more audio processing methods.
        self.res = []

    def is_fp_line(line):
        fp_category_list = ['FP', 'I', 'N', 'Q']
        elements = line.strip().split(' ')
        res = True
        for ele in elements:
            if ele not in fp_category_list:
                res = False
                break
        return res

    # TODO: adjust idx judgment rule
    def addfp(self, voice_output_dir, prosody, raw_metafile_lines):

        fp_category_list = ['FP', 'I', 'N']

        f = open(prosody)
        prosody_lines = f.readlines()
        f.close()

        idx = ''
        fp = ''
        fp_label_dict = {}
        i = 0
        while i < len(prosody_lines):
            if len(prosody_lines[i].strip().split('\t')) == 2:
                idx = prosody_lines[i].strip().split('\t')[0]
                i += 1
            else:
                fp_enable = is_fp_line(prosody_lines[i])
                if fp_enable:
                    fp = prosody_lines[i].strip().split('\t')[0].split(' ')
                    for label in fp:
                        if label not in fp_category_list:
                            logging.warning('fp label not in fp_category_list')
                            break
                    i += 4
                else:
                    fp = [
                        'N' for _ in range(
                            len(prosody_lines[i].strip().split('\t')
                                [0].replace('/ ', '').replace('. ', '').split(
                                    ' ')))
                    ]
                    i += 1
                fp_label_dict[idx] = fp

        fpadd_metafile = os.path.join(voice_output_dir, 'fpadd_metafile.txt')
        f_out = open(fpadd_metafile, 'w')
        for line in raw_metafile_lines:
            tokens = line.strip().split('\t')
            if len(tokens) == 2:
                uttname = tokens[0]
                symbol_sequences = tokens[1].split(' ')

                error_flag = False
                idx = 0
                out_str = uttname + '\t'

                for this_symbol_sequence in symbol_sequences:
                    emotion = this_symbol_sequence.split('$')[4]
                    this_symbol_sequence = this_symbol_sequence.replace(
                        emotion, 'emotion_neutral')

                    if idx < len(fp_label_dict[uttname]):
                        if fp_label_dict[uttname][idx] == 'FP':
                            if 'none' not in this_symbol_sequence:
                                this_symbol_sequence = this_symbol_sequence.replace(
                                    'emotion_neutral', 'emotion_disgust')
                        syllable_label = this_symbol_sequence.split('$')[2]
                        if syllable_label == 's_both' or syllable_label == 's_end':
                            idx += 1
                    elif idx > len(fp_label_dict[uttname]):
                        logging.warning(uttname + ' not match')
                        error_flag = True
                    out_str = out_str + this_symbol_sequence + ' '

                # if idx != len(fp_label_dict[uttname]):
                #     logging.warning(
                #         "{} length mismatch, length: {} ".format(
                #             idx, len(fp_label_dict[uttname])
                #         )
                #     )

                if not error_flag:
                    f_out.write(out_str.strip() + '\n')
        f_out.close()
        return fpadd_metafile

    def removefp(self, voice_output_dir, fpadd_metafile, raw_metafile_lines):

        f = open(fpadd_metafile)
        fpadd_metafile_lines = f.readlines()
        f.close()

        fprm_metafile = os.path.join(voice_output_dir, 'fprm_metafile.txt')
        f_out = open(fprm_metafile, 'w')
        for i in range(len(raw_metafile_lines)):
            tokens = raw_metafile_lines[i].strip().split('\t')
            symbol_sequences = tokens[1].split(' ')
            fpadd_tokens = fpadd_metafile_lines[i].strip().split('\t')
            fpadd_symbol_sequences = fpadd_tokens[1].split(' ')

            error_flag = False
            out_str = tokens[0] + '\t'
            idx = 0
            length = len(symbol_sequences)
            while idx < length:
                if '$emotion_disgust' in fpadd_symbol_sequences[idx]:
                    if idx + 1 < length and 'none' in fpadd_symbol_sequences[
                            idx + 1]:
                        idx = idx + 2
                    else:
                        idx = idx + 1
                    continue
                out_str = out_str + symbol_sequences[idx] + ' '
                idx = idx + 1

            if not error_flag:
                f_out.write(out_str.strip() + '\n')
        f_out.close()

    def process(self, voice_output_dir, prosody, raw_metafile):

        with open(raw_metafile, 'r') as f:
            lines = f.readlines()
        random.shuffle(lines)

        fpadd_metafile = self.addfp(voice_output_dir, prosody, lines)
        self.removefp(voice_output_dir, fpadd_metafile, lines)
