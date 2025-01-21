from argparse import Action, ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import fields
from typing import List


class CliArgumentParser(ArgumentParser):
    """ Argument Parser to define and parse command-line args for training.

    Args:
        training_args: dict or list of dict which defines different
            paramters for training.
    """

    def __init__(self, training_args=None, **kwargs):
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
