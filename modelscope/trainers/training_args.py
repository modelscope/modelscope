# Copyright (c) Alibaba, Inc. and its affiliates.

import dataclasses
from argparse import Action, ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Any, Dict, List, Union

from addict import Dict as Adict


@dataclasses.dataclass
class ArgAttr():
    """ Attributes for each arg

    Args:
        cfg_node_name (str or list[str]): if set empty, it means a normal arg for argparse, otherwise it means
            this arg value correspond to those nodes in configuration file, and will replace them for training.
        default:  default value for current argument.
        type:  type for current argument.
        choices (list of str): choices of value for this argument.
        help (str): help str for this argument.

    Examples:
    ```python
    # define argument train_batch_size which corresponds to train.dataloader.batch_size_per_gpu
    training_args = Adict(
        train_batch_size=ArgAttr(
            'train.dataloader.batch_size_per_gpu',
            default=16,
            type=int,
            help='training batch size')
    )

    # num_classes which will modify three places in configuration
    training_args = Adict(
    num_classes = ArgAttr(
        ['model.mm_model.head.num_classes',
         'model.mm_model.train_cfg.augments.0.num_classes',
         'model.mm_model.train_cfg.augments.1.num_classes'],
        type=int,
        help='number of classes')
    )
    ```
    # a normal argument which has no relation with configuration
    training_args = Adict(
        local_rank = ArgAttr(
            '',
            default=1,
            type=int,
            help='local rank for current training process')
        )

    """
    cfg_node_name: Union[str, List[str]] = ''
    default: Any = None
    type: type = None
    choices: List[str] = None
    help: str = ''


training_args = Adict(
    train_batch_size=ArgAttr(
        'train.dataloader.batch_size_per_gpu',
        default=16,
        type=int,
        help='training batch size'),
    train_data_worker=ArgAttr(
        'train.dataloader.workers_per_gpu',
        default=8,
        type=int,
        help='number of data worker used for training'),
    eval_batch_size=ArgAttr(
        'evaluation.dataloader.batch_size_per_gpu',
        default=16,
        type=int,
        help='training batch size'),
    max_epochs=ArgAttr(
        'train.max_epochs',
        default=10,
        type=int,
        help='max number of training epoch'),
    work_dir=ArgAttr(
        'train.work_dir',
        default='./work_dir',
        type=str,
        help='training directory to save models and training logs'),
    lr=ArgAttr(
        'train.optimizer.lr',
        default=0.001,
        type=float,
        help='initial learning rate'),
    optimizer=ArgAttr(
        'train.optimizer.type',
        default='SGD',
        type=str,
        choices=[
            'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD',
            'RMSprop', 'Rprop'
            'SGD'
        ],
        help='optimizer type'),
    local_rank=ArgAttr(
        '', default=0, type=int, help='local rank for this process'))


class CliArgumentParser(ArgumentParser):
    """ Argument Parser to define and parse command-line args for training.

    Args:
        arg_dict (dict of `ArgAttr` or list of them): dict or list of dict which defines different
            paramters for training.
    """

    def __init__(self, arg_dict: Union[Dict[str, ArgAttr],
                                       List[Dict[str, ArgAttr]]], **kwargs):
        if 'formatter_class' not in kwargs:
            kwargs['formatter_class'] = ArgumentDefaultsHelpFormatter
        super().__init__(**kwargs)
        self.arg_dict = arg_dict if isinstance(
            arg_dict, Dict) else self._join_args(arg_dict)
        self.define_args()

    def _join_args(self, arg_dict_list: List[Dict[str, ArgAttr]]):
        total_args = arg_dict_list[0].copy()
        for args in arg_dict_list[1:]:
            total_args.update(args)
        return total_args

    def define_args(self):
        for arg_name, arg_attr in self.arg_dict.items():
            name = f'--{arg_name}'
            kwargs = dict(type=arg_attr.type, help=arg_attr.help)
            if arg_attr.default is not None:
                kwargs['default'] = arg_attr.default
            else:
                kwargs['required'] = True

            if arg_attr.choices is not None:
                kwargs['choices'] = arg_attr.choices

            kwargs['action'] = SingleAction
            self.add_argument(name, **kwargs)

    def get_cfg_dict(self, args=None):
        """
        Args:
            args (default None):
                List of strings to parse. The default is taken from sys.argv. (same as argparse.ArgumentParser)

        Returns:
            cfg_dict (dict of config): each key is a config node name such as 'train.max_epochs', this cfg_dict
                should be used with function `cfg.merge_from_dict` to update config object.
        """
        self.args, remainning = self.parse_known_args(args)
        args_dict = vars(self.args)
        cfg_dict = {}
        for k, v in args_dict.items():
            if k not in self.arg_dict or self.arg_dict[k].cfg_node_name == '':
                continue
            cfg_node = self.arg_dict[k].cfg_node_name
            if isinstance(cfg_node, list):
                for node in cfg_node:
                    cfg_dict[node] = v
            else:
                cfg_dict[cfg_node] = v

        return cfg_dict


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
                string.count('[') == string.count(']')), \
                f'Imbalanced brackets exist in {string}'
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
