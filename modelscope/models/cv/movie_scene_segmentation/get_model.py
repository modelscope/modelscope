# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# Github: https://github.com/kakaobrain/bassl
# ------------------------------------------------------------------------------------

from .utils.shot_encoder import resnet50
from .utils.trn import TransformerCRN


def get_shot_encoder(cfg):
    name = cfg['model']['shot_encoder']['name']
    shot_encoder_args = cfg['model']['shot_encoder'][name]
    if name == 'resnet':
        depth = shot_encoder_args['depth']
        if depth == 50:
            shot_encoder = resnet50(**shot_encoder_args['params'], )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return shot_encoder


def get_contextual_relation_network(cfg):
    crn = None

    if cfg['model']['contextual_relation_network']['enabled']:
        name = cfg['model']['contextual_relation_network']['name']
        crn_args = cfg['model']['contextual_relation_network']['params'][name]
        if name == 'trn':
            sampling_name = cfg['model']['loss']['sampling_method']['name']
            crn_args['neighbor_size'] = (
                2 * cfg['model']['loss']['sampling_method']['params']
                [sampling_name]['neighbor_size'])
            crn = TransformerCRN(crn_args)
        else:
            raise NotImplementedError

    return crn


__all__ = ['get_shot_encoder', 'get_contextual_relation_network']
