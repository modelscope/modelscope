# Copyright (c) Alibaba, Inc. and its affiliates.

import contextlib
import hashlib
import os
import pickle
import random
import re
import shutil
import tempfile
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from types import FunctionType
from typing import Any, Dict, Union

import json
import numpy as np
import torch
import torch.optim
from torch import nn

from modelscope.utils.service_utils import NumpyEncoder


class RegressTool:
    """This class is used to stop inference/training results from changing by some unaware affections by unittests.

    Firstly, run a baseline test to create a result file, then changes can be observed between
    the latest version and the baseline file.
    """

    def __init__(self,
                 baseline: bool = None,
                 store_func: FunctionType = None,
                 load_func: FunctionType = None):
        """A func to store the baseline file and a func to load the baseline file.
        """
        self.baseline = baseline
        self.store_func = store_func
        self.load_func = load_func
        print(f'Current working dir is: {Path.cwd()}')

    def store(self, local, remote):
        if self.store_func is not None:
            self.store_func(local, remote)
        else:
            path = os.path.abspath(
                os.path.join(Path.cwd(), 'data', 'test', 'regression'))
            os.makedirs(path, exist_ok=True)
            shutil.copy(local, os.path.join(path, remote))

    def load(self, local, remote):
        if self.load_func is not None:
            self.load_func(local, remote)
        else:
            path = os.path.abspath(
                os.path.join(Path.cwd(), 'data', 'test', 'regression'))
            baseline = os.path.join(path, remote)
            if not os.path.exists(baseline):
                raise ValueError(f'base line file {baseline} not exist')
            print(
                f'local file found:{baseline}, md5:{hashlib.md5(open(baseline,"rb").read()).hexdigest()}'
            )
            if os.path.exists(local):
                os.remove(local)
            os.symlink(baseline, local, target_is_directory=False)

    @contextlib.contextmanager
    def monitor_module_single_forward(self,
                                      module: nn.Module,
                                      file_name: str,
                                      compare_fn=None,
                                      **kwargs):
        """Monitor a pytorch module in a single forward.

        Args:
            module: A torch module
            file_name: The file_name to store or load file
            compare_fn: A custom fn used to compare the results manually.

        >>> def compare_fn(v1, v2, key, type):
        >>>     return None

        v1 is the baseline value
        v2 is the value of current version
        key is the key of submodules
        type is in one of 'input', 'output'

            kwargs:
            atol: The absolute gap between two np arrays.
            rtol: The relative gap between two np arrays.
        """
        baseline = os.getenv('REGRESSION_BASELINE')
        if baseline is None or self.baseline is None:
            yield
            return

        baseline = self.baseline
        io_json = {}
        absolute_path = f'./{file_name}.bin'
        if not isinstance(module, nn.Module):
            assert hasattr(module, 'model')
            module = module.model

        hack_forward(module, file_name, io_json)
        intercept_module(module, io_json)
        yield
        hack_forward(module, None, None, restore=True)
        intercept_module(module, None, restore=True)
        if baseline:
            with open(absolute_path, 'wb') as f:
                pickle.dump(io_json, f)
            self.store(absolute_path, f'{file_name}.bin')
            os.remove(absolute_path)
        else:
            name = os.path.basename(absolute_path)
            baseline = os.path.join(tempfile.gettempdir(), name)
            self.load(baseline, name)
            with open(baseline, 'rb') as f:
                base = pickle.load(f)

            class SafeNumpyEncoder(NumpyEncoder):

                def default(self, obj):
                    try:
                        return super().default(obj)
                    except Exception:
                        print(
                            f'Type {obj.__class__} cannot be serialized and printed'
                        )
                        return None

            print(f'baseline: {json.dumps(base, cls=SafeNumpyEncoder)}')
            print(f'latest  : {json.dumps(io_json, cls=SafeNumpyEncoder)}')
            if not compare_io_and_print(base, io_json, compare_fn, **kwargs):
                raise ValueError('Result not match!')

    @contextlib.contextmanager
    def monitor_module_train(self,
                             trainer: Union[Dict, Any],
                             file_name,
                             level='config',
                             compare_fn=None,
                             ignore_keys=None,
                             compare_random=True,
                             reset_dropout=True,
                             lazy_stop_callback=None,
                             **kwargs):
        """Monitor a pytorch module's backward data and cfg data within a step of the optimizer.

        This is usually useful when you try to change some dangerous code
        which has the risk of affecting the training loop.

        Args:
            trainer: A dict or an object contains the model/optimizer/lr_scheduler
            file_name: The file_name to store or load file
            level: The regression level.
            'strict' for matching every single tensor.
                     Please make sure the parameters of head are fixed
                     and the drop-out rate is zero.
            'config' for matching the initial config, like cfg file, optimizer param_groups,
                     lr_scheduler params and the random seed.
            'metric' for compare the best metrics in the evaluation loop.
            compare_fn: A custom fn used to compare the results manually.
            ignore_keys: The keys to ignore of the named_parameters.
            compare_random: If to compare random setttings, default True.
            reset_dropout: Reset all dropout modules to 0.0.
            lazy_stop_callback: A callback passed in, when the moniting is over, this callback will be called.
            kwargs:
            atol: The absolute gap between two np arrays.
            rtol: The relative gap between two np arrays.

        >>> def compare_fn(v1, v2, key, type):
        >>>     return None

        v1 is the baseline value
        v2 is the value of current version
        key is the key of modules/parameters
        type is in one of 'input', 'output', 'backward', 'optimizer', 'lr_scheduler', 'cfg', 'state'
        """
        baseline = os.getenv('REGRESSION_BASELINE')
        if baseline is None or self.baseline is None:
            yield
            return

        baseline = self.baseline

        io_json = {}
        bw_json = {}
        absolute_path = f'./{file_name}.bin'

        if level == 'strict':
            print(
                "[Important] The level of regression is 'strict', please make sure your model's parameters are "
                'fixed and all drop-out rates have been set to zero.')

        assert hasattr(
            trainer, 'model') or 'model' in trainer, 'model must be in trainer'
        module = trainer['model'] if isinstance(trainer,
                                                dict) else trainer.model
        if not isinstance(module, nn.Module):
            assert hasattr(module, 'model')
            module = module.model

        assert hasattr(
            trainer, 'optimizer'
        ) or 'optimizer' in trainer, 'optimizer must be in trainer'
        assert hasattr(
            trainer, 'lr_scheduler'
        ) or 'lr_scheduler' in trainer, 'lr_scheduler must be in trainer'
        optimizer: torch.optim.Optimizer = trainer['optimizer'] if isinstance(
            trainer, dict) else trainer.optimizer
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = trainer['lr_scheduler'] if isinstance(trainer, dict) \
            else trainer.lr_scheduler
        torch_state = numpify_tensor_nested(torch.get_rng_state())
        np_state = np.random.get_state()
        random_seed = random.getstate()
        seed = trainer._seed if hasattr(
            trainer,
            '_seed') else trainer.seed if hasattr(trainer, 'seed') else None

        if reset_dropout:
            with torch.no_grad():

                def reinit_dropout(_module):
                    for name, submodule in _module.named_children():
                        if isinstance(submodule, torch.nn.Dropout):
                            setattr(_module, name, torch.nn.Dropout(0.))
                        else:
                            reinit_dropout(submodule)

                reinit_dropout(module)

        if level == 'strict':
            hack_forward(module, file_name, io_json)
            intercept_module(module, io_json)
        hack_backward(
            module, optimizer, bw_json, lazy_stop_callback=lazy_stop_callback)
        yield
        hack_backward(module, optimizer, None, restore=True)
        if level == 'strict':
            hack_forward(module, None, None, restore=True)
            intercept_module(module, None, restore=True)

        optimizer_dict = optimizer.state_dict()
        optimizer_dict.pop('state', None)
        summary = {
            'forward': io_json,
            'backward': bw_json,
            'optimizer': {
                'type': optimizer.__class__.__name__,
                'defaults': optimizer.defaults,
                'state_dict': optimizer_dict
            },
            'lr_scheduler': {
                'type': lr_scheduler.__class__.__name__,
                'state_dict': lr_scheduler.state_dict()
            },
            'cfg': trainer.cfg.to_dict() if hasattr(trainer, 'cfg') else None,
            'state': {
                'torch_state': torch_state,
                'np_state': np_state,
                'random_seed': random_seed,
                'seed': seed,
            }
        }

        if baseline:
            with open(absolute_path, 'wb') as f:
                pickle.dump(summary, f)
            self.store(absolute_path, f'{file_name}.bin')
            os.remove(absolute_path)
        else:
            name = os.path.basename(absolute_path)
            baseline = os.path.join(tempfile.gettempdir(), name)
            self.load(baseline, name)
            with open(baseline, 'rb') as f:
                baseline_json = pickle.load(f)

            if level == 'strict' and not compare_io_and_print(
                    baseline_json['forward'], io_json, compare_fn, **kwargs):
                raise RuntimeError('Forward not match!')
            if not compare_backward_and_print(
                    baseline_json['backward'],
                    bw_json,
                    compare_fn=compare_fn,
                    ignore_keys=ignore_keys,
                    level=level,
                    **kwargs):
                raise RuntimeError('Backward not match!')
            cfg_opt1 = {
                'optimizer': baseline_json['optimizer'],
                'lr_scheduler': baseline_json['lr_scheduler'],
                'cfg': baseline_json['cfg'],
                'state': None if not compare_random else baseline_json['state']
            }
            cfg_opt2 = {
                'optimizer': summary['optimizer'],
                'lr_scheduler': summary['lr_scheduler'],
                'cfg': summary['cfg'],
                'state': None if not compare_random else summary['state']
            }
            if not compare_cfg_and_optimizers(cfg_opt1, cfg_opt2, compare_fn,
                                              **kwargs):
                raise RuntimeError('Cfg or optimizers not match!')


class MsRegressTool(RegressTool):

    class EarlyStopError(Exception):
        pass

    @contextlib.contextmanager
    def monitor_ms_train(self,
                         trainer,
                         file_name,
                         level='config',
                         compare_fn=None,
                         ignore_keys=None,
                         compare_random=True,
                         lazy_stop_callback=None,
                         **kwargs):

        if lazy_stop_callback is None:

            def lazy_stop_callback():

                from modelscope.trainers.hooks.hook import Hook, Priority

                class EarlyStopHook(Hook):
                    PRIORITY = Priority.VERY_LOW

                    def after_iter(self, trainer):
                        raise MsRegressTool.EarlyStopError('Test finished.')

                trainer.register_hook(EarlyStopHook())

        def _train_loop(trainer, *args_train, **kwargs_train):
            with self.monitor_module_train(
                    trainer,
                    file_name,
                    level,
                    compare_fn=compare_fn,
                    ignore_keys=ignore_keys,
                    compare_random=compare_random,
                    lazy_stop_callback=lazy_stop_callback,
                    **kwargs):
                try:
                    return trainer.train_loop_origin(*args_train,
                                                     **kwargs_train)
                except MsRegressTool.EarlyStopError:
                    pass

        trainer.train_loop_origin, trainer.train_loop = \
            trainer.train_loop, type(trainer.train_loop)(_train_loop, trainer)
        yield


def compare_module(module1: nn.Module, module2: nn.Module):
    for p1, p2 in zip(module1.parameters(), module2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def numpify_tensor_nested(tensors, reduction=None, clip_value=10000):
    try:
        from modelscope.outputs import ModelOutputBase
    except ImportError:
        ModelOutputBase = dict
    "Numpify `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (Mapping, ModelOutputBase)):
        return OrderedDict({
            k: numpify_tensor_nested(t, reduction, clip_value)
            for k, t in tensors.items()
        })
    if isinstance(tensors, list):
        return list(
            numpify_tensor_nested(t, reduction, clip_value) for t in tensors)
    if isinstance(tensors, tuple):
        return tuple(
            numpify_tensor_nested(t, reduction, clip_value) for t in tensors)
    if isinstance(tensors, torch.Tensor):
        t: np.ndarray = tensors.cpu().numpy()
        if clip_value is not None:
            t = np.where(t > clip_value, clip_value, t)
            t = np.where(t < -clip_value, -clip_value, t)
        if reduction == 'sum':
            return t.sum(dtype=np.float)
        elif reduction == 'mean':
            return t.mean(dtype=np.float)
        return t
    return tensors


def detach_tensor_nested(tensors):
    try:
        from modelscope.outputs import ModelOutputBase
    except ImportError:
        ModelOutputBase = dict
    "Detach `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (Mapping, ModelOutputBase)):
        return OrderedDict(
            {k: detach_tensor_nested(t)
             for k, t in tensors.items()})
    if isinstance(tensors, list):
        return list(detach_tensor_nested(t) for t in tensors)
    if isinstance(tensors, tuple):
        return tuple(detach_tensor_nested(t) for t in tensors)
    if isinstance(tensors, torch.Tensor):
        return tensors.detach()
    return tensors


def hack_forward(module: nn.Module,
                 name,
                 io_json,
                 restore=False,
                 keep_tensors=False):

    def _forward(self, *args, **kwargs):
        ret = self.forward_origin(*args, **kwargs)
        if keep_tensors:
            args = numpify_tensor_nested(detach_tensor_nested(args))
            kwargs = numpify_tensor_nested(detach_tensor_nested(kwargs))
            output = numpify_tensor_nested(detach_tensor_nested(ret))
        else:
            args = {
                'sum':
                numpify_tensor_nested(
                    detach_tensor_nested(args), reduction='sum'),
                'mean':
                numpify_tensor_nested(
                    detach_tensor_nested(args), reduction='mean'),
            }
            kwargs = {
                'sum':
                numpify_tensor_nested(
                    detach_tensor_nested(kwargs), reduction='sum'),
                'mean':
                numpify_tensor_nested(
                    detach_tensor_nested(kwargs), reduction='mean'),
            }
            output = {
                'sum':
                numpify_tensor_nested(
                    detach_tensor_nested(ret), reduction='sum'),
                'mean':
                numpify_tensor_nested(
                    detach_tensor_nested(ret), reduction='mean'),
            }

        io_json[name] = {
            'input': {
                'args': args,
                'kwargs': kwargs,
            },
            'output': output,
        }
        return ret

    if not restore and not hasattr(module, 'forward_origin'):
        module.forward_origin, module.forward = module.forward, type(
            module.forward)(_forward, module)
    if restore and hasattr(module, 'forward_origin'):
        module.forward = module.forward_origin
        del module.forward_origin


def hack_backward(module: nn.Module,
                  optimizer,
                  io_json,
                  restore=False,
                  lazy_stop_callback=None):

    def _step(self, *args, **kwargs):
        for name, param in module.named_parameters():
            io_json[name] = {
                'data': {
                    'sum':
                    numpify_tensor_nested(
                        detach_tensor_nested(param.data), reduction='sum'),
                    'mean':
                    numpify_tensor_nested(
                        detach_tensor_nested(param.data), reduction='mean'),
                },
                'grad': {
                    'sum':
                    numpify_tensor_nested(
                        detach_tensor_nested(param.grad), reduction='sum'),
                    'mean':
                    numpify_tensor_nested(
                        detach_tensor_nested(param.grad), reduction='mean'),
                }
            }
        ret = self.step_origin(*args, **kwargs)
        for name, param in module.named_parameters():
            io_json[name]['data_after'] = {
                'sum':
                numpify_tensor_nested(
                    detach_tensor_nested(param.data), reduction='sum'),
                'mean':
                numpify_tensor_nested(
                    detach_tensor_nested(param.data), reduction='mean'),
            }
        if lazy_stop_callback is not None:
            lazy_stop_callback()
        return ret

    if not restore and not hasattr(optimizer, 'step_origin'):
        optimizer.step_origin, optimizer.step = optimizer.step, type(
            optimizer.state_dict)(_step, optimizer)
    if restore and hasattr(optimizer, 'step_origin'):
        optimizer.step = optimizer.step_origin
        del optimizer.step_origin


def intercept_module(module: nn.Module,
                     io_json,
                     parent_name=None,
                     restore=False):
    for name, module in module.named_children():
        full_name = parent_name + '.' + name if parent_name is not None else name
        hack_forward(module, full_name, io_json, restore)
        intercept_module(module, io_json, full_name, restore)


def compare_arguments_nested(print_content,
                             arg1,
                             arg2,
                             rtol=1.e-3,
                             atol=1.e-8,
                             ignore_unknown_type=True):
    type1 = type(arg1)
    type2 = type(arg2)
    if type1.__name__ != type2.__name__:
        if print_content is not None:
            print(
                f'{print_content}, type not equal:{type1.__name__} and {type2.__name__}'
            )
        return False

    if arg1 is None:
        return True
    elif isinstance(arg1, (int, str, bool, np.bool, np.integer, np.str)):
        if arg1 != arg2:
            if print_content is not None:
                print(f'{print_content}, arg1:{arg1}, arg2:{arg2}')
            return False
        return True
    elif isinstance(arg1, (float, np.floating)):
        if not np.isclose(arg1, arg2, rtol=rtol, atol=atol, equal_nan=True):
            if print_content is not None:
                print(f'{print_content}, arg1:{arg1}, arg2:{arg2}')
            return False
        return True
    elif isinstance(arg1, (tuple, list)):
        if len(arg1) != len(arg2):
            if print_content is not None:
                print(
                    f'{print_content}, length is not equal:{len(arg1)}, {len(arg2)}'
                )
            return False
        if not all([
                compare_arguments_nested(
                    None, sub_arg1, sub_arg2, rtol=rtol, atol=atol)
                for sub_arg1, sub_arg2 in zip(arg1, arg2)
        ]):
            if print_content is not None:
                print(f'{print_content}')
            return False
        return True
    elif isinstance(arg1, Mapping):
        keys1 = arg1.keys()
        keys2 = arg2.keys()
        if len(keys1) != len(keys2):
            if print_content is not None:
                print(
                    f'{print_content}, key length is not equal:{len(keys1)}, {len(keys2)}'
                )
            return False
        if len(set(keys1) - set(keys2)) > 0:
            if print_content is not None:
                print(f'{print_content}, key diff:{set(keys1) - set(keys2)}')
            return False
        if not all([
                compare_arguments_nested(
                    None, arg1[key], arg2[key], rtol=rtol, atol=atol)
                for key in keys1
        ]):
            if print_content is not None:
                print(f'{print_content}')
            return False
        return True
    elif isinstance(arg1, np.ndarray):
        arg1 = np.where(np.equal(arg1, None), np.NaN,
                        arg1).astype(dtype=np.float)
        arg2 = np.where(np.equal(arg2, None), np.NaN,
                        arg2).astype(dtype=np.float)
        if not all(
                np.isclose(arg1, arg2, rtol=rtol, atol=atol,
                           equal_nan=True).flatten()):
            if print_content is not None:
                print(f'{print_content}')
            return False
        return True
    else:
        if ignore_unknown_type:
            return True
        else:
            raise ValueError(f'type not supported: {type1}')


def compare_io_and_print(baseline_json, io_json, compare_fn=None, **kwargs):
    if compare_fn is None:

        def compare_fn(*args, **kwargs):
            return None

    keys1 = set(baseline_json.keys())
    keys2 = set(io_json.keys())
    added = keys1 - keys2
    removed = keys2 - keys1
    print(f'unmatched keys: {added}, {removed}')
    shared_keys = keys1.intersection(keys2)
    match = True
    for key in shared_keys:
        v1 = baseline_json[key]
        v2 = io_json[key]

        v1input = numpify_tensor_nested(v1['input'])
        v2input = numpify_tensor_nested(v2['input'])
        res = compare_fn(v1input, v2input, key, 'input')
        if res is not None:
            print(
                f'input of {key} compared with user compare_fn with result:{res}\n'
            )
            match = match and res
        else:
            match = compare_arguments_nested(
                f'unmatched module {key} input args', v1input['args'],
                v2input['args'], **kwargs) and match
            match = compare_arguments_nested(
                f'unmatched module {key} input kwargs', v1input['kwargs'],
                v2input['kwargs'], **kwargs) and match
        v1output = numpify_tensor_nested(v1['output'])
        v2output = numpify_tensor_nested(v2['output'])
        res = compare_fn(v1output, v2output, key, 'output')
        if res is not None:
            print(
                f'output of {key} compared with user compare_fn with result:{res}\n'
            )
            match = match and res
        else:
            match = compare_arguments_nested(
                f'unmatched module {key} outputs',
                arg1=v1output,
                arg2=v2output,
                **kwargs) and match
    return match


def compare_backward_and_print(baseline_json,
                               bw_json,
                               level,
                               ignore_keys=None,
                               compare_fn=None,
                               **kwargs):
    if compare_fn is None:

        def compare_fn(*args, **kwargs):
            return None

    keys1 = set(baseline_json.keys())
    keys2 = set(bw_json.keys())
    added = keys1 - keys2
    removed = keys2 - keys1
    print(f'unmatched backward keys: {added}, {removed}')
    shared_keys = keys1.intersection(keys2)
    match = True
    for key in shared_keys:
        if ignore_keys is not None and key in ignore_keys:
            continue

        res = compare_fn(baseline_json[key], bw_json[key], key, 'backward')
        if res is not None:
            print(f'backward data of {key} compared with '
                  f'user compare_fn with result:{res}\n')
            match = match and res
        else:
            data1, grad1, data_after1 = baseline_json[key][
                'data'], baseline_json[key]['grad'], baseline_json[key][
                    'data_after']
            data2, grad2, data_after2 = bw_json[key]['data'], bw_json[key][
                'grad'], bw_json[key]['data_after']
            match = compare_arguments_nested(
                f'unmatched module {key} tensor data',
                arg1=data1,
                arg2=data2,
                **kwargs) and match
            if level == 'strict':
                match = compare_arguments_nested(
                    f'unmatched module {key} grad data',
                    arg1=grad1,
                    arg2=grad2,
                    **kwargs) and match
                match = compare_arguments_nested(
                    f'unmatched module {key} data after step', data_after1,
                    data_after2, **kwargs) and match
    return match


def compare_cfg_and_optimizers(baseline_json,
                               cfg_json,
                               compare_fn=None,
                               **kwargs):
    if compare_fn is None:

        def compare_fn(*args, **kwargs):
            return None

    optimizer1, lr_scheduler1, cfg1, state1 = baseline_json[
        'optimizer'], baseline_json['lr_scheduler'], baseline_json[
            'cfg'], baseline_json['state']
    optimizer2, lr_scheduler2, cfg2, state2 = cfg_json['optimizer'], cfg_json[
        'lr_scheduler'], cfg_json['cfg'], baseline_json['state']

    match = True
    res = compare_fn(optimizer1, optimizer2, None, 'optimizer')
    if res is not None:
        print(f'optimizer compared with user compare_fn with result:{res}\n')
        match = match and res
    else:
        if optimizer1['type'] != optimizer2['type']:
            print(
                f"Optimizer type not equal:{optimizer1['type']} and {optimizer2['type']}"
            )
        match = compare_arguments_nested(
            'unmatched optimizer defaults', optimizer1['defaults'],
            optimizer2['defaults'], **kwargs) and match
        match = compare_arguments_nested(
            'unmatched optimizer state_dict', optimizer1['state_dict'],
            optimizer2['state_dict'], **kwargs) and match

    res = compare_fn(lr_scheduler1, lr_scheduler2, None, 'lr_scheduler')
    if res is not None:
        print(
            f'lr_scheduler compared with user compare_fn with result:{res}\n')
        match = match and res
    else:
        if lr_scheduler1['type'] != lr_scheduler2['type']:
            print(
                f"Optimizer type not equal:{lr_scheduler1['type']} and {lr_scheduler2['type']}"
            )
        match = compare_arguments_nested(
            'unmatched lr_scheduler state_dict', lr_scheduler1['state_dict'],
            lr_scheduler2['state_dict'], **kwargs) and match

    res = compare_fn(cfg1, cfg2, None, 'cfg')
    if res is not None:
        print(f'cfg compared with user compare_fn with result:{res}\n')
        match = match and res
    else:
        match = compare_arguments_nested(
            'unmatched cfg', arg1=cfg1, arg2=cfg2, **kwargs) and match

    res = compare_fn(state1, state2, None, 'state')
    if res is not None:
        print(
            f'random state compared with user compare_fn with result:{res}\n')
        match = match and res
    else:
        match = compare_arguments_nested('unmatched random state', state1,
                                         state2, **kwargs) and match

    return match


class IgnoreKeyFn:

    def __init__(self, keys):
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys if isinstance(keys, list) else []

    def __call__(self, v1output, v2output, key, type):
        for _key in self.keys:
            pattern = re.compile(_key)
            if key is not None and pattern.fullmatch(key):
                return True
        return None
