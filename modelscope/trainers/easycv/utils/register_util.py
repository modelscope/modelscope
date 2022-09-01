# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import logging

from modelscope.trainers.hooks import HOOKS
from modelscope.trainers.parallel.builder import PARALLEL


def register_parallel():
    from mmcv.parallel import MMDistributedDataParallel, MMDataParallel

    PARALLEL.register_module(
        module_name='MMDistributedDataParallel',
        module_cls=MMDistributedDataParallel)
    PARALLEL.register_module(
        module_name='MMDataParallel', module_cls=MMDataParallel)


def register_hook_to_ms(hook_name, logger=None):
    """Register EasyCV hook to ModelScope."""
    from easycv.hooks import HOOKS as _EV_HOOKS

    if hook_name not in _EV_HOOKS._module_dict:
        raise ValueError(
            f'Not found hook "{hook_name}" in EasyCV hook registries!')

    obj = _EV_HOOKS._module_dict[hook_name]
    HOOKS.register_module(module_name=hook_name, module_cls=obj)

    log_str = f'Register hook "{hook_name}" to modelscope hooks.'
    logger.info(log_str) if logger is not None else logging.info(log_str)


def register_part_mmcv_hooks_to_ms():
    """Register required mmcv hooks to ModelScope.
    Currently we only registered all lr scheduler hooks in EasyCV and mmcv.
    Please refer to:
        EasyCV: https://github.com/alibaba/EasyCV/blob/master/easycv/hooks/lr_update_hook.py
        mmcv: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py
    """
    from mmcv.runner.hooks import lr_updater
    from mmcv.runner.hooks import HOOKS as _MMCV_HOOKS
    from easycv.hooks import StepFixCosineAnnealingLrUpdaterHook, YOLOXLrUpdaterHook
    from easycv.hooks.logger import PreLoggerHook

    mmcv_hooks_in_easycv = [('StepFixCosineAnnealingLrUpdaterHook',
                             StepFixCosineAnnealingLrUpdaterHook),
                            ('YOLOXLrUpdaterHook', YOLOXLrUpdaterHook),
                            ('PreLoggerHook', PreLoggerHook)]

    members = inspect.getmembers(lr_updater)
    members.extend(mmcv_hooks_in_easycv)

    for name, obj in members:
        if name in _MMCV_HOOKS._module_dict:
            HOOKS.register_module(
                module_name=name,
                module_cls=obj,
            )
