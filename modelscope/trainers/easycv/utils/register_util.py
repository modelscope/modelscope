# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import logging

from modelscope.trainers.hooks import HOOKS
from modelscope.trainers.parallel.builder import PARALLEL
from modelscope.utils.registry import default_group


class _RegisterManager:

    def __init__(self):
        self.registries = {}

    def add(self, module, name, group_key=default_group):
        if module.name not in self.registries:
            self.registries[module.name] = {}
        if group_key not in self.registries[module.name]:
            self.registries[module.name][group_key] = []

        self.registries[module.name][group_key].append(name)

    def exists(self, module, name, group_key=default_group):
        if self.registries.get(module.name, None) is None:
            return False
        if self.registries[module.name].get(group_key, None) is None:
            return False
        if name in self.registries[module.name][group_key]:
            return True

        return False


_dynamic_register = _RegisterManager()


def register_parallel():
    from mmcv.parallel import MMDistributedDataParallel, MMDataParallel

    mmddp = 'MMDistributedDataParallel'
    mmdp = 'MMDataParallel'

    if not _dynamic_register.exists(PARALLEL, mmddp):
        _dynamic_register.add(PARALLEL, mmddp)
        PARALLEL.register_module(
            module_name=mmddp, module_cls=MMDistributedDataParallel)
    if not _dynamic_register.exists(PARALLEL, mmdp):
        _dynamic_register.add(PARALLEL, mmdp)
        PARALLEL.register_module(module_name=mmdp, module_cls=MMDataParallel)


def register_hook_to_ms(hook_name, logger=None):
    """Register EasyCV hook to ModelScope."""
    from easycv.hooks import HOOKS as _EV_HOOKS

    if hook_name not in _EV_HOOKS._module_dict:
        raise ValueError(
            f'Not found hook "{hook_name}" in EasyCV hook registries!')

    if _dynamic_register.exists(HOOKS, hook_name):
        return
    _dynamic_register.add(HOOKS, hook_name)

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

    mmcv_hooks_in_easycv = [('StepFixCosineAnnealingLrUpdaterHook',
                             StepFixCosineAnnealingLrUpdaterHook),
                            ('YOLOXLrUpdaterHook', YOLOXLrUpdaterHook)]

    members = inspect.getmembers(lr_updater)
    members.extend(mmcv_hooks_in_easycv)

    for name, obj in members:
        if name in _MMCV_HOOKS._module_dict:
            if _dynamic_register.exists(HOOKS, name):
                continue
            _dynamic_register.add(HOOKS, name)
            HOOKS.register_module(
                module_name=name,
                module_cls=obj,
            )
