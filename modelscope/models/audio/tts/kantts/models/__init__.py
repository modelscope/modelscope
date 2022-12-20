# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
from torch.nn.parallel import DistributedDataParallel

import modelscope.models.audio.tts.kantts.train.scheduler as kantts_scheduler
from modelscope.models.audio.tts.kantts.utils.ling_unit.ling_unit import \
    get_fpdict
from .hifigan import (Generator, MultiPeriodDiscriminator,
                      MultiScaleDiscriminator, MultiSpecDiscriminator)
from .pqmf import PQMF
from .sambert.kantts_sambert import KanTtsSAMBERT, KanTtsTextsyBERT


def optimizer_builder(model_params, opt_name, opt_params):
    opt_cls = getattr(torch.optim, opt_name)
    optimizer = opt_cls(model_params, **opt_params)
    return optimizer


def scheduler_builder(optimizer, sche_name, sche_params):
    scheduler_cls = getattr(kantts_scheduler, sche_name)
    scheduler = scheduler_cls(optimizer, **sche_params)
    return scheduler


def hifigan_model_builder(config, device, rank, distributed):
    model = {}
    optimizer = {}
    scheduler = {}
    model['discriminator'] = {}
    optimizer['discriminator'] = {}
    scheduler['discriminator'] = {}
    for model_name in config['Model'].keys():
        if model_name == 'Generator':
            params = config['Model'][model_name]['params']
            model['generator'] = Generator(**params).to(device)
            optimizer['generator'] = optimizer_builder(
                model['generator'].parameters(),
                config['Model'][model_name]['optimizer'].get('type', 'Adam'),
                config['Model'][model_name]['optimizer'].get('params', {}),
            )
            scheduler['generator'] = scheduler_builder(
                optimizer['generator'],
                config['Model'][model_name]['scheduler'].get('type', 'StepLR'),
                config['Model'][model_name]['scheduler'].get('params', {}),
            )
        else:
            params = config['Model'][model_name]['params']
            model['discriminator'][model_name] = globals()[model_name](
                **params).to(device)
            optimizer['discriminator'][model_name] = optimizer_builder(
                model['discriminator'][model_name].parameters(),
                config['Model'][model_name]['optimizer'].get('type', 'Adam'),
                config['Model'][model_name]['optimizer'].get('params', {}),
            )
            scheduler['discriminator'][model_name] = scheduler_builder(
                optimizer['discriminator'][model_name],
                config['Model'][model_name]['scheduler'].get('type', 'StepLR'),
                config['Model'][model_name]['scheduler'].get('params', {}),
            )

    out_channels = config['Model']['Generator']['params']['out_channels']
    if out_channels > 1:
        model['pqmf'] = PQMF(
            subbands=out_channels, **config.get('pqmf', {})).to(device)

    # FIXME: pywavelets buffer leads to gradient error in DDP training
    # Solution: https://github.com/pytorch/pytorch/issues/22095
    if distributed:
        model['generator'] = DistributedDataParallel(
            model['generator'],
            device_ids=[rank],
            output_device=rank,
            broadcast_buffers=False,
        )
        for model_name in model['discriminator'].keys():
            model['discriminator'][model_name] = DistributedDataParallel(
                model['discriminator'][model_name],
                device_ids=[rank],
                output_device=rank,
                broadcast_buffers=False,
            )

    return model, optimizer, scheduler


def sambert_model_builder(config, device, rank, distributed):
    model = {}
    optimizer = {}
    scheduler = {}

    model['KanTtsSAMBERT'] = KanTtsSAMBERT(
        config['Model']['KanTtsSAMBERT']['params']).to(device)

    fp_enable = config['Model']['KanTtsSAMBERT']['params'].get('FP', False)
    if fp_enable:
        fp_dict = {
            k: torch.from_numpy(v).long().unsqueeze(0).to(device)
            for k, v in get_fpdict(config).items()
        }
        model['KanTtsSAMBERT'].fp_dict = fp_dict

    optimizer['KanTtsSAMBERT'] = optimizer_builder(
        model['KanTtsSAMBERT'].parameters(),
        config['Model']['KanTtsSAMBERT']['optimizer'].get('type', 'Adam'),
        config['Model']['KanTtsSAMBERT']['optimizer'].get('params', {}),
    )
    scheduler['KanTtsSAMBERT'] = scheduler_builder(
        optimizer['KanTtsSAMBERT'],
        config['Model']['KanTtsSAMBERT']['scheduler'].get('type', 'StepLR'),
        config['Model']['KanTtsSAMBERT']['scheduler'].get('params', {}),
    )

    if distributed:
        model['KanTtsSAMBERT'] = DistributedDataParallel(
            model['KanTtsSAMBERT'], device_ids=[rank], output_device=rank)

    return model, optimizer, scheduler


def sybert_model_builder(config, device, rank, distributed):
    model = {}
    optimizer = {}
    scheduler = {}

    model['KanTtsTextsyBERT'] = KanTtsTextsyBERT(
        config['Model']['KanTtsTextsyBERT']['params']).to(device)
    optimizer['KanTtsTextsyBERT'] = optimizer_builder(
        model['KanTtsTextsyBERT'].parameters(),
        config['Model']['KanTtsTextsyBERT']['optimizer'].get('type', 'Adam'),
        config['Model']['KanTtsTextsyBERT']['optimizer'].get('params', {}),
    )
    scheduler['KanTtsTextsyBERT'] = scheduler_builder(
        optimizer['KanTtsTextsyBERT'],
        config['Model']['KanTtsTextsyBERT']['scheduler'].get('type', 'StepLR'),
        config['Model']['KanTtsTextsyBERT']['scheduler'].get('params', {}),
    )

    if distributed:
        model['KanTtsTextsyBERT'] = DistributedDataParallel(
            model['KanTtsTextsyBERT'], device_ids=[rank], output_device=rank)

    return model, optimizer, scheduler


model_dict = {
    'hifigan': hifigan_model_builder,
    'sambert': sambert_model_builder,
    'sybert': sybert_model_builder,
}


def model_builder(config, device='cpu', rank=0, distributed=False):
    builder_func = model_dict[config['model_type']]
    model, optimizer, scheduler = builder_func(config, device, rank,
                                               distributed)
    return model, optimizer, scheduler
