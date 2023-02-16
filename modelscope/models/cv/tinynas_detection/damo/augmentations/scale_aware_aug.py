# This file mainly comes from
# https://github.com/dvlab-research/SA-AutoAug/blob/master/FCOS/fcos_core/augmentations/scale_aware_aug.py
# Copyright © Alibaba, Inc. and its affiliates.

import copy

from .box_level_augs.box_level_augs import Box_augs
from .box_level_augs.color_augs import color_aug_func
from .box_level_augs.geometric_augs import geometric_aug_func


class SA_Aug(object):

    def __init__(self, iters_per_epoch, start_epoch, total_epochs,
                 no_aug_epochs, batch_size, num_gpus, num_workers, sada_cfg):

        autoaug_list = sada_cfg.autoaug_params
        num_policies = sada_cfg.num_subpolicies
        scale_splits = sada_cfg.scale_splits
        box_prob = sada_cfg.box_prob

        self.batch_size = batch_size / num_gpus
        self.num_workers = num_workers
        self.max_iters = (total_epochs - no_aug_epochs) * iters_per_epoch
        self.count = start_epoch * iters_per_epoch
        if self.num_workers == 0:
            self.num_workers += 1

        box_aug_list = autoaug_list[4:]
        color_aug_types = list(color_aug_func.keys())
        geometric_aug_types = list(geometric_aug_func.keys())
        policies = []
        for i in range(num_policies):
            _start_pos = i * 6
            sub_policy = [
                (
                    color_aug_types[box_aug_list[_start_pos + 0]
                                    % len(color_aug_types)],
                    box_aug_list[_start_pos + 1] * 0.1,
                    box_aug_list[_start_pos + 2],
                ),  # box_color policy
                (geometric_aug_types[box_aug_list[_start_pos + 3]
                                     % len(geometric_aug_types)],
                 box_aug_list[_start_pos + 4] * 0.1,
                 box_aug_list[_start_pos + 5])
            ]  # box_geometric policy
            policies.append(sub_policy)

        _start_pos = num_policies * 6
        scale_ratios = {
            'area': [
                box_aug_list[_start_pos + 0], box_aug_list[_start_pos + 1],
                box_aug_list[_start_pos + 2]
            ],
            'prob': [
                box_aug_list[_start_pos + 3], box_aug_list[_start_pos + 4],
                box_aug_list[_start_pos + 5]
            ]
        }

        box_augs_dict = {'policies': policies, 'scale_ratios': scale_ratios}

        self.box_augs = Box_augs(
            box_augs_dict=box_augs_dict,
            max_iters=self.max_iters,
            scale_splits=scale_splits,
            box_prob=box_prob)

    def __call__(self, tensor, target):
        iteration = self.count // self.batch_size * self.num_workers
        tensor = copy.deepcopy(tensor)
        target = copy.deepcopy(target)
        tensor, target = self.box_augs(tensor, target, iteration=iteration)

        self.count += 1

        return tensor, target
