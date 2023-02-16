# Copyright (c) Alibaba, Inc. and its affiliates.

import re
from argparse import Action, ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass, field, fields
from functools import partial
from typing import Any, Dict, List, Tuple, Union

from modelscope.trainers.default_config import DEFAULT_CONFIG
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.hub import read_config


def get_flatten_value(config: Config, metadata: Dict, exclusions=None):
    cfg_node = metadata['cfg_node']
    if exclusions is None:
        exclusions = []

    values = config.safe_get(cfg_node)
    if isinstance(values, dict):
        param_map = []
        for key, value in values.items():
            if key in exclusions or not isinstance(value,
                                                   (str, int, float, bool)):
                continue
            value = add_quotes_for_str(value)
            param_map.append(f'{key}={value}')
        return ','.join(param_map)
    else:
        return values


def set_flatten_value(config: Config, values: Union[str, List[str]],
                      metadata: Dict):
    cfg_node = metadata['cfg_node']
    if values is None:
        return config

    pairs = values.split(',') if isinstance(values, str) else values
    for kv in pairs:
        if len(kv.strip()) == 0:
            continue
        key, value = kv.split('=')
        value = parse_value(value)
        config.merge_from_dict({cfg_node + '.' + key: value})
    return config


def get_base_hook_args(config: Config, metadata: Dict):
    cfg_node = metadata['cfg_node']
    hook_type = metadata['hook_type']
    key = metadata['key']
    value = config.safe_get(cfg_node)
    if value is None:
        return get_hook_param(config, hook_type, key)
    else:
        return True if key == 'type' else value


def set_base_hook_args(config: Config, value: Any, metadata: Dict):
    cfg_node = metadata['cfg_node']
    hook_type = metadata['hook_type']
    key = metadata['key']
    if 'hooks' in config.train:
        config.train.hooks = [
            hook for hook in config.train.hooks if hook['type'] != hook_type
        ]
    if key == 'type':
        if value and config.safe_get(cfg_node) is None:
            config.merge_from_dict({cfg_node: {}})
    else:
        config.merge_from_dict({cfg_node: value})


def get_strategy(config: Config,
                 metadata: Dict,
                 value_pair: Tuple[str] = ('by_epoch', 'by_step')):
    flag = get_base_hook_args(config, metadata)
    if flag is None:
        return None
    return value_pair[0] if flag else value_pair[1]


def set_strategy(config: Config,
                 value: Any,
                 metadata: Dict,
                 value_pair: Tuple[str] = ('by_epoch', 'by_step')):
    set_base_hook_args(config, value == value_pair[0], metadata)


def get_hook_param(config, hook_type: str, key='type'):
    hooks = config.safe_get('train.hooks', [])
    _hooks = list(filter(lambda hook: hook['type'] == hook_type, hooks))
    if key == 'type':
        return len(_hooks) > 0
    elif len(_hooks) > 0:
        return getattr(_hooks[0], key, None)
    return None


def add_quotes_for_str(value: Union[str, float, bool, None]) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    else:
        return str(value)


def parse_value(value: str) -> Union[str, float, bool, None]:
    const_map = {
        'True': True,
        'true': True,
        'False': False,
        'false': False,
        'None': None,
        'none': None,
        'null': None
    }
    if value in const_map:
        return const_map[value]
    elif '"' in value or "'" in value:
        return value.replace('"', '').replace("'", '')
    elif re.match(r'^\d+$', value):
        return int(value)
    elif re.match(r'[+-]?(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?',
                  value):
        return float(value)
    else:
        return value


@dataclass
class TrainingArgs:
    model: str = field(
        default=None, metadata={
            'help': 'A model id or model dir',
        })

    seed: int = field(
        default=42, metadata={
            'help': 'The random seed',
        })

    task: str = field(
        default=None,
        metadata={
            'help': 'The task code to be used',
            'cfg_node': 'task'
        })

    dataset_name: str = field(
        default=None, metadata={
            'help': 'The dataset name',
        })

    subset_name: str = field(
        default=None, metadata={
            'help': 'The subset name of the dataset',
        })

    train_dataset_name: str = field(
        default=None, metadata={
            'help': 'The train dataset name',
        })

    val_dataset_name: str = field(
        default=None, metadata={
            'help': 'The validation dataset name',
        })

    per_device_train_batch_size: int = field(
        default=None,
        metadata={
            'cfg_node': 'train.dataloader.batch_size_per_gpu',
            'help': 'The training batch size per GPU',
        })

    train_data_worker: int = field(
        default=None,
        metadata={
            'cfg_node': 'train.dataloader.workers_per_gpu',
            'help': 'The number of data workers for train dataloader',
        })

    train_shuffle: bool = field(
        default=None,
        metadata={
            'cfg_node': 'train.dataloader.shuffle',
            'help': 'Shuffle the train dataset or not',
        })

    per_device_eval_batch_size: int = field(
        default=None,
        metadata={
            'cfg_node': 'evaluation.dataloader.batch_size_per_gpu',
            'help': 'The eval batch size per GPU',
        })

    eval_data_worker: int = field(
        default=None,
        metadata={
            'cfg_node': 'evaluation.dataloader.workers_per_gpu',
            'help': 'The number of data workers for eval dataloader',
        })

    eval_shuffle: bool = field(
        default=None,
        metadata={
            'cfg_node': 'evaluation.dataloader.shuffle',
            'help': 'Shuffle the eval dataset or not',
        })

    max_epochs: int = field(
        default=None,
        metadata={
            'cfg_node': 'train.max_epochs',
            'help': 'The training epochs',
        })

    work_dir: str = field(
        default=None,
        metadata={
            'cfg_node': 'train.work_dir',
            'help': 'The training dir to save models and logs',
        })

    lr: float = field(
        default=None,
        metadata={
            'cfg_node': 'train.optimizer.lr',
            'help': 'The learning rate of the optimizer',
        })

    optimizer: str = field(
        default=None,
        metadata={
            'cfg_node': 'train.optimizer.type',
            'help': 'The optimizer type',
        })

    optimizer_params: str = field(
        default=None,
        metadata={
            'cfg_node':
            'train.optimizer',
            'cfg_getter':
            partial(get_flatten_value, exclusions=['type', 'lr', 'options']),
            'cfg_setter':
            set_flatten_value,
            'help':
            'The optimizer init params except `lr`',
        })

    lr_scheduler_params: str = field(
        default=None,
        metadata={
            'cfg_node':
            'train.lr_scheduler',
            'cfg_getter':
            partial(get_flatten_value, exclusions=['type', 'lr', 'options']),
            'cfg_setter':
            set_flatten_value,
            'help':
            'The lr_scheduler init params',
        })

    local_rank: int = field(
        default=0, metadata={
            'help': 'The training local rank',
        })

    save_ckpt: bool = field(
        default=True,
        metadata={
            'help':
            'Periodically save checkpoint when True, corresponding to CheckpointHook',
            'cfg_node': 'train.checkpoint.period',
            'hook_type': 'CheckpointHook',
            'key': 'type',
            'cfg_getter': get_base_hook_args,
            'cfg_setter': set_base_hook_args,
        })

    save_ckpt_best: bool = field(
        default=None,
        metadata={
            'help':
            'Save best checkpoint when True, corresponding to BestCkptSaverHook',
            'cfg_node': 'train.checkpoint.best',
            'hook_type': 'BestCkptSaverHook',
            'key': 'type',
            'cfg_getter': get_base_hook_args,
            'cfg_setter': set_base_hook_args,
        })

    evaluate: bool = field(
        default=True,
        metadata={
            'help': 'Evaluate when True, corresponding to EvaluationHook',
            'cfg_node': 'evaluation.period',
            'hook_type': 'EvaluationHook',
            'key': 'type',
            'cfg_getter': get_base_hook_args,
            'cfg_setter': set_base_hook_args,
        })

    save_ckpt_strategy: str = field(
        default=None,
        metadata={
            'help': 'Periodically save checkpoint by epoch or by step'
            'use with `CheckpointHook`, can be `by_epoch` or `by_step`',
            'cfg_node': 'train.checkpoint.period.by_epoch',
            'hook_type': 'CheckpointHook',
            'key': 'by_epoch',
            'choices': ['by_epoch', 'by_step'],
            'cfg_getter': get_strategy,
            'cfg_setter': set_strategy,
        })

    save_ckpt_best_strategy: str = field(
        default=None,
        metadata={
            'help': 'Save best checkpoint by epoch or by step'
            'use with `BestCkptSaverHook`, can be `by_epoch` or `by_step`',
            'cfg_node': 'train.checkpoint.best.by_epoch',
            'hook_type': 'BestCkptSaverHook',
            'key': 'by_epoch',
            'choices': ['by_epoch', 'by_step'],
            'cfg_getter': get_strategy,
            'cfg_setter': set_strategy,
        })

    ckpt_period_interval: int = field(
        default=1,
        metadata={
            'help':
            'The interval of epoch or iter of saving checkpoint period',
            'cfg_node': 'train.checkpoint.period.interval',
            'hook_type': 'CheckpointHook',
            'key': 'interval',
            'cfg_getter': get_base_hook_args,
            'cfg_setter': set_base_hook_args,
        })

    ckpt_best_interval: int = field(
        default=None,
        metadata={
            'help': 'The interval of epoch or iter of saving checkpoint best',
            'cfg_node': 'train.checkpoint.best.interval',
            'hook_type': 'BestCkptSaverHook',
            'key': 'interval',
            'cfg_getter': get_base_hook_args,
            'cfg_setter': set_base_hook_args,
        })

    metric_for_best_model: str = field(
        default=None,
        metadata={
            'help':
            'Which metric key to judge the checkpoint is better or not, use with `BestCkptSaverHook`, '
            'please make sure this key is returned by the `evaluation_metrics` classes',
            'cfg_node':
            'train.checkpoint.best.metric_key',
            'hook_type':
            'BestCkptSaverHook',
            'key':
            'metric_key',
            'cfg_getter':
            get_base_hook_args,
            'cfg_setter':
            set_base_hook_args,
        })

    metric_rule_for_best_model: str = field(
        default=None,
        metadata={
            'help':
            'Which rule to compare the value of `checkpoint_saving_metric`, '
            'use with `BestCkptSaverHook`, can be `max` or `min`',
            'cfg_node':
            'train.checkpoint.best.rule',
            'hook_type':
            'BestCkptSaverHook',
            'key':
            'rule',
            'cfg_getter':
            get_base_hook_args,
            'cfg_setter':
            set_base_hook_args,
        })

    save_ckpt_peroid_limit: int = field(
        default=None,
        metadata={
            'help':
            'The max saving number of checkpoint, older checkpoints will be deleted.',
            'cfg_node': 'train.checkpoint.period.max_checkpoint_num',
            'hook_type': 'CheckpointHook',
            'key': 'max_checkpoint_num',
            'cfg_getter': get_base_hook_args,
            'cfg_setter': set_base_hook_args,
        })

    save_ckpt_best_limit: int = field(
        default=None,
        metadata={
            'help':
            'The max saving number of checkpoint, worse checkpoints will be deleted.',
            'cfg_node': 'train.checkpoint.best.max_checkpoint_num',
            'hook_type': 'BestCkptSaverHook',
            'key': 'max_checkpoint_num',
            'cfg_getter': get_base_hook_args,
            'cfg_setter': set_base_hook_args,
        })

    logging_interval: int = field(
        default=None,
        metadata={
            'help': 'The interval of iter of logging information',
            'cfg_node': 'train.logging.interval',
            'hook_type': 'TextLoggerHook',
            'key': 'interval',
            'cfg_getter': get_base_hook_args,
            'cfg_setter': set_base_hook_args,
        })

    eval_strategy: str = field(
        default=None,
        metadata={
            'help': 'Evaluate model by epoch or by step'
            'use with `EvaluationHook`, can be `by_epoch` or `by_step`',
            'cfg_node': 'evaluation.period.by_epoch',
            'hook_type': 'EvaluationHook',
            'key': 'by_epoch',
            'choices': ['by_epoch', 'by_step'],
            'cfg_getter': get_strategy,
            'cfg_setter': set_strategy,
        })

    eval_interval: int = field(
        default=1,
        metadata={
            'help': 'Evaluation interval by epoch or iter',
            'cfg_node': 'evaluation.period.interval',
            'hook_type': 'EvaluationHook',
            'key': 'interval',
            'cfg_getter': get_base_hook_args,
            'cfg_setter': set_base_hook_args,
        })

    eval_metrics: str = field(
        default=None,
        metadata={
            'help': 'The metric module name used in evaluation',
            'cfg_node': 'evaluation.metrics'
        })

    @classmethod
    def from_cli(cls, parser_args=None, **extra_kwargs):
        """Construct a TrainingArg class by the parameters of CLI.

        Args:
            **extra_kwargs: Extra args which can be defined in code.

        Returns:
            The output TrainingArg class with the parameters from CLI.
        """
        self = cls(**extra_kwargs)
        parser = CliArgumentParser(self)
        args, unknown = parser.parse_known_args(parser_args)
        unknown = [item for item in unknown if item not in ('\\', '\n')]
        _unknown = {}
        for i in range(0, len(unknown), 2):
            _unknown[unknown[i].replace('-', '')] = parse_value(unknown[i + 1])
        cfg_dict = vars(args)

        if args.model is not None:
            try:
                cfg = read_config(args.model)
            except Exception as e:
                print('Read config failed with error:', e)
            else:
                cfg.merge_from_dict(_unknown)
                self = cls.from_config(cfg, **extra_kwargs)
        for key, value in cfg_dict.items():
            if key is not None and hasattr(self,
                                           key) and key in parser.manual_args:
                setattr(self, key, value)
        return self

    def to_args(self):
        """Convert the TrainingArg class to key-value pairs.

        Returns: The key-value pair.

        """
        _args = {}
        for f in fields(self):
            _args[f.name] = getattr(self, f.name)
        return _args

    @classmethod
    def from_config(cls, config=DEFAULT_CONFIG, **kwargs):
        """Construct the TrainingArg class by a `Config` class.

        Args:
            config: The Config class. By default, `DEFAULT_CONFIG` is used.
            **kwargs: Extra args which can be defined in code.

        Returns: The output TrainingArg class with the parameters from the config.

        """

        self = cls(**kwargs)
        for f in fields(self):
            if 'cfg_node' in f.metadata and getattr(self, f.name) is None:
                self._to_field(f, config)
        return self

    def _to_field(self, f, config):
        assert 'cfg_node' in f.metadata
        if 'cfg_getter' in f.metadata:
            cfg_getter = f.metadata['cfg_getter']
            setattr(self, f.name, cfg_getter(config, f.metadata))
        else:
            cfg_node = f.metadata['cfg_node']
            setattr(self, f.name, config.safe_get(cfg_node))

    def _to_config(self, f, config: Config):
        assert 'cfg_node' in f.metadata
        value = getattr(self, f.name)
        if 'cfg_setter' in f.metadata:
            cfg_setter = f.metadata['cfg_setter']
            config = cfg_setter(config, value, f.metadata)
        else:
            cfg_node = f.metadata['cfg_node']
            if isinstance(cfg_node, str):
                cfg_node = [cfg_node]
            for _node in cfg_node:
                config.merge_from_dict({_node: value})
        return config

    def __call__(self, cfg: Config):
        for f in fields(self):
            if 'cfg_node' not in f.metadata:
                continue

            value = getattr(self, f.name)
            if value is not None:
                self._to_config(f, cfg)
            else:
                self._to_field(f, cfg)
        return cfg


class CliArgumentParser(ArgumentParser):
    """ Argument Parser to define and parse command-line args for training.

    Args:
        training_args (TrainingArgs): dict or list of dict which defines different
            paramters for training.
    """

    def __init__(self, training_args: TrainingArgs = None, **kwargs):
        if 'formatter_class' not in kwargs:
            kwargs['formatter_class'] = ArgumentDefaultsHelpFormatter
        super().__init__(**kwargs)
        self.training_args = training_args
        self.define_args()

    def get_manual_args(self, args):
        return [arg[2:] for arg in args if arg.startswith('--')]

    def _parse_known_args(self, args: List = None, namespace=None):
        self.model_id = namespace.model if namespace is not None else None
        if '--model' in args:
            self.model_id = args[args.index('--model') + 1]
        self.manual_args = self.get_manual_args(args)
        return super()._parse_known_args(args, namespace)

    def print_help(self, file=None):
        config = DEFAULT_CONFIG
        if self.model_id is not None:
            try:
                config = read_config(self.model_id)
            except Exception as e:
                print('Read config failed with error:', e)

        if config is not None:
            for action_group in self._optionals._group_actions:
                if hasattr(self.training_args, action_group.dest):
                    value = getattr(self.training_args, action_group.dest)
                    f = {f.name: f
                         for f in fields(self.training_args)
                         }.get(action_group.dest)
                    if value is not None:
                        action_group.default = value
                    elif 'cfg_node' in f.metadata:
                        cfg_node = f.metadata['cfg_node']
                        if isinstance(cfg_node, str):
                            cfg_node = [cfg_node]

                        assert isinstance(cfg_node, (list, tuple))
                        if isinstance(cfg_node[0], str):
                            action_group.default = config.safe_get(cfg_node[0])
                        else:
                            action_group.default = cfg_node[0](config)
        return super().print_help(file)

    def define_args(self):
        if self.training_args is not None:
            for f in fields(self.training_args):
                arg_name = f.name
                arg_attr = getattr(self.training_args, f.name)
                name = f'--{arg_name}'
                kwargs = dict(type=f.type, help=f.metadata['help'])
                kwargs['default'] = arg_attr

                if 'choices' in f.metadata:
                    kwargs['choices'] = f.metadata['choices']

                kwargs['action'] = SingleAction
                self.add_argument(name, **kwargs)


class DictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def parse_int_float_bool_str(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return val.lower() == 'true'
        if val == 'None':
            return None
        return val

    @staticmethod
    def parse_iterable(val):
        """Parse iterable values in the string.
        All elements inside '()' or '[]' are treated as iterable values.
        Args:
            val (str): Value string.
        Returns:
            list | tuple: The expanded list or tuple from the string.
        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b'], 'c']
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.
            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count('(') == string.count(')')) and (
                string.count('[')
                == string.count(']')), f'Imbalanced brackets exist in {string}'
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if ((char == ',') and (pre.count('(') == pre.count(')'))
                        and (pre.count('[') == pre.count(']'))):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip('\'\"').replace(' ', '')
        is_tuple = False
        if val.startswith('(') and val.endswith(')'):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith('[') and val.endswith(']'):
            val = val[1:-1]
        elif ',' not in val:
            # val is a single value
            return DictAction.parse_int_float_bool_str(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction.parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1:]
        if is_tuple:
            values = tuple(values)
        return values

    def __call__(self, parser, namespace, values, option_string):
        options = {}
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            options[key] = self.parse_iterable(val)
        setattr(namespace, self.dest, options)


class SingleAction(DictAction):
    """ Argparse action to convert value to tuple or list or nested structure of
    list and tuple, i.e 'V1,V2,V3', or with explicit brackets, i.e. '[V1,V2,V3]'.
    It also support nested brackets to build list/tuple values. e.g. '[(V1,V2),(V3,V4)]'
    """

    def __call__(self, parser, namespace, value, option_string):
        if isinstance(value, str):
            setattr(namespace, self.dest, self.parse_iterable(value))
        else:
            setattr(namespace, self.dest, value)
