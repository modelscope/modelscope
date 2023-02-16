# Copyright (c) Alibaba, Inc. and its affiliates.

from functools import wraps
from inspect import signature


def type_assert(*ty_args, **ty_kwargs):
    """a decorator which is used to check the types of arguments in a function or class
    Examples:
        >>> @type_assert(str)
        ... def main(a: str, b: list):
        ...     print(a, b)
        >>> main(1)
        Argument a must be a str

        >>> @type_assert(str, (int, str))
        ... def main(a: str, b: int | str):
        ...     print(a, b)
        >>> main('1', [1])
        Argument b must be (<class 'int'>, <class 'str'>)

        >>> @type_assert(str, (int, str))
        ... class A:
        ...     def __init__(self, a: str, b: int | str)
        ...         print(a, b)
        >>> a = A('1', [1])
        Argument b must be (<class 'int'>, <class 'str'>)
    """

    def decorate(func):
        # If in optimized mode, disable type checking
        if not __debug__:
            return func

        # Map function argument names to supplied types
        sig = signature(func)
        bound_types = sig.bind_partial(*ty_args, **ty_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)
            # Enforce type assertions across supplied arguments
            for name, value in bound_values.arguments.items():
                if name in bound_types:
                    if not isinstance(value, bound_types[name]):
                        raise TypeError('Argument {} must be {}'.format(
                            name, bound_types[name]))
            return func(*args, **kwargs)

        return wrapper

    return decorate
