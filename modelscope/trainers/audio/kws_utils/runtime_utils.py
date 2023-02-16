import codecs
import os
import re
import stat
import sys
from collections import OrderedDict
from shutil import copyfile

import json

from modelscope.utils.logger import get_logger

logger = get_logger()


def make_runtime_res(model_dir, dest_path, kaldi_text, keywords):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    logger.info(f'making runtime resource in {dest_path} for {keywords}')

    # keywords split with ',', like 'keyword1,keyword2, ...'
    keywords_list = keywords.strip().replace(' ', '').split(',')

    kaldi_path = os.path.join(model_dir, 'train')
    kaldi_tool = os.path.join(model_dir, 'train/nnet-copy')
    kaldi_net = os.path.join(dest_path, 'kwsr.net')
    os.environ['PATH'] = f'{kaldi_path}:$PATH'
    os.environ['LD_LIBRARY_PATH'] = f'{kaldi_path}:$LD_LIBRARYPATH'
    assert os.path.exists(kaldi_tool)
    os.chmod(kaldi_tool, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
    os.system(f'{kaldi_tool} --binary=true {kaldi_text} {kaldi_net}')

    copyfile(
        os.path.join(model_dir, 'kwsr.ccl'),
        os.path.join(dest_path, 'kwsr.ccl'))
    copyfile(
        os.path.join(model_dir, 'kwsr.cfg'),
        os.path.join(dest_path, 'kwsr.cfg'))
    copyfile(
        os.path.join(model_dir, 'kwsr.gbg'),
        os.path.join(dest_path, 'kwsr.gbg'))
    copyfile(
        os.path.join(model_dir, 'kwsr.lex'),
        os.path.join(dest_path, 'kwsr.lex'))
    copyfile(
        os.path.join(model_dir, 'kwsr.mdl'),
        os.path.join(dest_path, 'kwsr.mdl'))
    copyfile(
        os.path.join(model_dir, 'kwsr.mvn'),
        os.path.join(dest_path, 'kwsr.mvn'))
    copyfile(
        os.path.join(model_dir, 'kwsr.phn'),
        os.path.join(dest_path, 'kwsr.phn'))
    copyfile(
        os.path.join(model_dir, 'kwsr.tree'),
        os.path.join(dest_path, 'kwsr.tree'))
    copyfile(
        os.path.join(model_dir, 'kwsr.prior'),
        os.path.join(dest_path, 'kwsr.prior'))

    # build keywords grammar
    keywords_grammar = os.path.join(dest_path, 'keywords.json')

    keywords_root = {}
    keywords_root['word_list'] = []
    for keyword in keywords_list:
        one_dict = OrderedDict()
        one_dict['name'] = keyword
        one_dict['type'] = 'wakeup'
        one_dict['activation'] = True
        one_dict['is_main'] = True
        one_dict['lm_boost'] = 0.0
        one_dict['am_boost'] = 0.0
        one_dict['threshold1'] = 0.0
        one_dict['threshold2'] = -1.0
        one_dict['subseg_threshold'] = -0.6
        one_dict['high_threshold'] = 90.0
        one_dict['min_dur'] = 0.4
        one_dict['max_dur'] = 2.5
        one_dict['cc_name'] = 'commoncc'
        keywords_root['word_list'].append(one_dict)

    with codecs.open(keywords_grammar, 'w', encoding='utf-8') as fh:
        json.dump(keywords_root, fh, indent=4, ensure_ascii=False)
        fh.close()
