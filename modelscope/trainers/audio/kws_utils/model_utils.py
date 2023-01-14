# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: di.wu@mobvoi.com (DI WU)

import glob
import os
import re
from shutil import copyfile

import numpy as np
import torch
import yaml

from modelscope.utils.checkpoint import load_checkpoint, save_checkpoint
from modelscope.utils.logger import get_logger

logger = get_logger()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def average_model(**kwargs):
    assert kwargs.get('dst_model', None) is not None, \
        'Please config param: dst_model, to save averaged model'
    dst_model = kwargs['dst_model']

    assert kwargs.get('src_path', None) is not None, \
        'Please config param: src_path, path of checkpoints to be averaged'
    src_path = kwargs['src_path']

    val_best = kwargs.get('val_best',
                          'True')  # average with best loss or final models

    avg_num = kwargs.get('avg_num', 5)  # nums for averaging model

    min_epoch = kwargs.get('min_epoch',
                           5)  # min epoch used for averaging model
    max_epoch = kwargs.get('max_epoch',
                           65536)  # max epoch used for averaging model

    val_scores = []
    if val_best:
        yamls = glob.glob('{}/[!config]*.yaml'.format(src_path))
        for y in yamls:
            with open(y, 'r') as f:
                dic_yaml = yaml.load(f, Loader=yaml.FullLoader)
                print(y, dic_yaml)
                loss = dic_yaml['cv_loss']
                epoch = dic_yaml['epoch']
                if epoch >= min_epoch and epoch <= max_epoch:
                    val_scores += [[epoch, loss]]
        val_scores = np.array(val_scores)
        sort_idx = np.argsort(val_scores[:, -1])
        sorted_val_scores = val_scores[sort_idx][::1]
        logger.info('best val scores = ' + str(sorted_val_scores[:avg_num, 1]))
        logger.info('selected epochs = '
                    + str(sorted_val_scores[:avg_num, 0].astype(np.int64)))
        path_list = [
            src_path + '/{}.pt'.format(int(epoch))
            for epoch in sorted_val_scores[:avg_num, 0]
        ]
    else:
        path_list = glob.glob('{}/[!avg][!final]*.pt'.format(src_path))
        path_list = sorted(path_list, key=os.path.getmtime)
        path_list = path_list[-avg_num:]

    logger.info(path_list)
    avg = None

    # assert num == len(path_list)
    if avg_num > len(path_list):
        logger.info(
            'insufficient epochs for averaging, exist num:{}, need:{}'.format(
                len(path_list), avg_num))
        logger.info('select epoch on best val:{}'.format(path_list[0]))
        path_list = [path_list[0]]

    for path in path_list:
        logger.info('Processing {}'.format(path))
        states = torch.load(path, map_location=torch.device('cpu'))
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            # pytorch 1.6 use true_divide instead of /=
            # avg[k] = torch.true_divide(avg[k], num)
            avg[k] = torch.true_divide(avg[k], len(path_list))
    logger.info('Saving to {}'.format(dst_model))
    torch.save(avg, dst_model)

    return dst_model


def convert_to_kaldi(
    model: torch.nn.Module,
    network_file: str,
    model_dir: str,
):
    copyfile(network_file, os.path.join(model_dir, 'origin.torch.pt'))
    load_checkpoint(network_file, model)

    kaldi_text = os.path.join(model_dir, 'convert.kaldi.txt')
    with open(kaldi_text, 'w', encoding='utf8') as fout:
        nnet_desp = model.to_kaldi_net()
        fout.write(nnet_desp)
    fout.close()

    return kaldi_text


def convert_to_pytorch(
    model: torch.nn.Module,
    network_file: str,
    model_dir: str,
):
    num_params = count_parameters(model)
    logger.info('the number of model params: {}'.format(num_params))

    copyfile(network_file, os.path.join(model_dir, 'origin.kaldi.txt'))
    model.to_pytorch_net(network_file)

    save_model_path = os.path.join(model_dir, 'convert.torch.pt')
    save_checkpoint(model, save_model_path, None, None, None, False)

    logger.info('convert torch format back to kaldi for recheck...')
    kaldi_text = os.path.join(model_dir, 'convert.kaldi.txt')
    with open(kaldi_text, 'w', encoding='utf8') as fout:
        nnet_desp = model.to_kaldi_net()
        fout.write(nnet_desp)
    fout.close()

    return save_model_path
