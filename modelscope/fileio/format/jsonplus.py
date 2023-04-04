# Modify from: https://github.com/randomir/jsonplus/blob/master/python/jsonplus/__init__.py
"""Custom datatypes (like datetime) serialization to/from JSON."""

# TODO: handle environments without threads
# (Python compiled without thread support)

import simplejson as json
from operator import attrgetter
from sortedcontainers import SortedList
from datetime import datetime, timedelta, date, time
from dateutil.parser import parse as parse_datetime
from functools import wraps, partial
from operator import methodcaller
from decimal import Decimal
from fractions import Fraction
from collections import namedtuple
import threading
import uuid
import numpy as np

try:
    from moneyed import Money, Currency
except ImportError:
    # defer failing to actual (de-)serialization
    pass

__all__ = ["loads", "dumps", "pretty",
           "json_loads", "json_dumps", "json_prettydump",
           "encoder", "decoder"]


# Should we aim for the *exact* reproduction of Python types,
# or for maximum *compatibility* when (de-)serializing?
#
# By default, we prefer the exactness of reproduction.
# For example, `tuple`, `namedtuple`, `Decimal`, or `datetime` will all be
# restored to the appropriate type (same as the starting type -- even the
# custom class for the `namedtuple` is recreated).
# When compatible coding if turned on, we shall fallback to standard JSON
# types, and values from the example above will be serialized as
# `list` (`Array`), `dict` (`Object`), `Number` and `ISO8601 timestamp string`,
# respectively.
#
# Please note:
#  - `compat` mode is lossy -- `namedtuple` serialized as `dict`/`Object`
#    can never be deserialized as `namedtuple`.
#  - `exact` mode is verbose -- and if you have a standard JSON decoder on
#    the other end, all that additional type info is useless/discared.
#
# To switch between representation styles, use `jsonplus.prefer(coding)`,
# where `coding` is `jsonplus.EXACT` or `jsonplus.COMPAT`. Another way, maybe
# simpler, is to use `jsonplus.prefer_exact()` and `jsonplus.prefer_compat()`.
#
# The preference is stored thread-local.

EXACT = 1
COMPAT = 2
CODING_DEFAULT = EXACT

_local = threading.local()

def prefer(coding):
    _local.coding = coding

def prefer_exact():
    prefer(EXACT)

def prefer_compat():
    prefer(COMPAT)


def getattrs(value, attrs):
    """Helper function that extracts a list of attributes from
    `value` object in a `dict`/mapping of (attr, value[attr]).

    Args:
        value (object):
            Any Python object upon which `getattr` can act.

        attrs (iterable):
            Any iterable containing attribute names for extract.

    Returns:
        `dict` of attr -> val mappings.

    Example:
        >>> getattrs(complex(2,3), ['imag', 'real'])
        {'imag': 3.0, 'real': 2.0}
    """
    return dict([(attr, getattr(value, attr)) for attr in attrs])


def kwargified(constructor):
    """Function decorator that wraps a function receiving
    keyword arguments into a function receiving a dictionary
    of arguments.

    Example:
        @kwargified
        def test(a=1, b=2):
            return a + b

        >>> test({'b': 3})
        4
    """
    @wraps(constructor)
    def kwargs_constructor(kwargs):
        return constructor(**kwargs)
    return kwargs_constructor


_PredicatedEncoder = namedtuple('_PredicatedEncoder',
                                'priority predicate encoder typename')

def encoder(classname, predicate=None, priority=None, exact=True):
    """A decorator for registering a new encoder for object type
    defined either by a `classname`, or detected via `predicate`.

    Predicates are tested according to priority (low to high),
    but always before classname.

    Args:
        classname (str):
            Classname of the object serialized, equal to
            ``type(obj).__name__``.

        predicate (callable, default=None):
            A predicate for testing if object is of certain type.
            The predicate shall receive a single argument, the object
            being encoded, and it has to return a boolean `True/False`.
            See examples below.

        priority (int, default=None):
            Predicate priority. If undefined, encoder is added at
            the end, with lowest priority.

        exact (bool, default=True):
            Determines the kind of encoder registered, an exact
            (default), or a compact representation encoder.

    Examples:
        @encoder('mytype')
        def mytype_exact_encoder(myobj):
            return myobj.to_json()

        Functional discriminator usage is appropriate for serialization
        of objects with a different classname, but which can be encoded
        with the same encoder:

        @encoder('BaseClass', lambda obj: isinstance(obj, BaseClass))
        def all_derived_classes_encoder(derived):
            return derived.base_encoder()
    """
    if exact:
        subregistry = _encode_handlers['exact']
    else:
        subregistry = _encode_handlers['compat']

    # if priority undefined, set it to lowest
    if priority is None:
        if len(subregistry['predicate']) > 0:
            priority = subregistry['predicate'][-1].priority + 100
        else:
            priority = 1000

    def _decorator(f):
        if predicate:
            subregistry['predicate'].add(
                _PredicatedEncoder(priority, predicate, f, classname))
        else:
            subregistry['classname'].setdefault(classname, f)
        return f

    return _decorator


def _json_default_exact(obj):
    """Serialization handlers for types unsupported by `simplejson` 
    that try to preserve the exact data types.
    """

    # first try predicate-based encoders
    for handler in _encode_handlers['exact']['predicate']:
        if handler.predicate(obj):
            return {"__class__": handler.typename,
                    "__value__": handler.encoder(obj)}

    # then classname-based
    classname = type(obj).__name__
    if classname in _encode_handlers['exact']['classname']:
        return {"__class__": classname,
                "__value__": _encode_handlers['exact']['classname'][classname](obj)}

    raise TypeError(repr(obj) + " is not JSON serializable")


def _json_default_compat(obj):
    """Serialization handlers that try to dump objects in
    compatibility mode. Similar to above.
    """
    for handler in _encode_handlers['compat']['predicate']:
        if handler.predicate(obj):
            return handler.encoder(obj)
    classname = type(obj).__name__
    if classname in _encode_handlers['compat']['classname']:
        return _encode_handlers['compat']['classname'][classname](obj)
    raise TypeError(repr(obj) + " is not JSON serializable")


def decoder(classname):
    """A decorator for registering a new decoder for `classname`.
    Only ``exact`` decoders can be registered, since it is an assumption
    the ``compat`` mode serializes to standard JSON.

    Example:
        @decoder('mytype')
        def mytype_decoder(value):
            return mytype(value, reconstruct=True)
    """
    def _decorator(f):
        _decode_handlers.setdefault(classname, f)
    return _decorator


def _json_object_hook(dict):
    """Deserialization handlers for types unsupported by `simplejson`.
    """
    classname = dict.get('__class__')
    if classname:
        constructor = _decode_handlers.get(classname)
        value = dict.get('__value__')
        if constructor:
            return constructor(value)
        raise TypeError("Unknown class: '%s'" % classname)
    return dict



def _encoder_default_args(kw):
    """Shape default arguments for encoding functions."""
    
    # manual override of the preferred coding with `exact=False`
    if kw.pop('exact', getattr(_local, 'coding', CODING_DEFAULT) == EXACT):
        # settings necessary for the "exact coding"
        kw.update({
            'default': _json_default_exact,
            'use_decimal': False,           # don't encode `Decimal` as JSON's `Number`
            'tuple_as_array': False,        # don't encode `tuple` as `Array`
            'namedtuple_as_object': False   # don't call `_asdict` on `namedtuple`
        })
    else:
        # settings for the "compatibility coding"
        kw.update({
            'default': _json_default_compat,
            'ignore_nan': True      # be compliant with the ECMA-262 specification:
                                    # serialize nan/inf as null
        })

    # NOTE: if called from ``simplejson.dumps()`` with ``cls=JSONEncoder``,
    # we will receive all kw set to simplejson defaults -- and our defaults for
    # ``separators`` and ``for_json`` will not be applied. In contrast, they
    # are applied when called from ``jsonplus.dumps()``, unless user explicitly
    # sets some of those.
    # This causes inconsistent behaviour between ``dumps()`` and ``JSONEncoder()``.

    # prefer compact json repr
    kw.setdefault('separators', (',', ':'))

    # allow objects to provide json serialization on its behalf
    kw.setdefault('for_json', True)


def _decoder_default_args(kw):
    """Shape default arguments for decoding functions."""

    kw.update({'object_hook': _json_object_hook})



class JSONEncoder(json.JSONEncoder):
    def __init__(self, **kw):
        """Constructor for simplejson.JSONEncoder, with defaults overriden
        for jsonplus.
        """
        _encoder_default_args(kw)
        super(JSONEncoder, self).__init__(**kw)


class JSONDecoder(json.JSONDecoder):
    def __init__(self, **kw):
        """Constructor for simplejson.JSONDecoder, with defaults overriden
        for jsonplus.
        """
        _decoder_default_args(kw)
        super(JSONDecoder, self).__init__(**kw)



def dumps(*pa, **kw):
    _encoder_default_args(kw)
    return json.dumps(*pa, **kw)


def loads(*pa, **kw):
    _decoder_default_args(kw)
    return json.loads(*pa, **kw)


def pretty(x, sort_keys=True, indent=4*' ', separators=(',', ': '), **kw):
    kw.setdefault('sort_keys', sort_keys)
    kw.setdefault('indent', indent)
    kw.setdefault('separators', separators)
    return dumps(x, **kw)



json_dumps = dumps
json_loads = loads
json_prettydump = pretty


def np_to_list(value):
    return value.tolist()


def generic_to_item(value):
    return value.item()


_encode_handlers = {
    'exact': {
        'classname': {
            'datetime': methodcaller('isoformat'),
            'date': methodcaller('isoformat'),
            'time': methodcaller('isoformat'),
            'timedelta': partial(getattrs, attrs=['days', 'seconds', 'microseconds']),
            'tuple': list,
            'set': list,
            'ndarray': np_to_list,
            'float16': generic_to_item,
            'float32': generic_to_item,
            'frozenset': list,
            'complex': partial(getattrs, attrs=['real', 'imag']),
            'Decimal': str,
            'Fraction': partial(getattrs, attrs=['numerator', 'denominator']),
            'UUID': partial(getattrs, attrs=['hex']),
            'Money': partial(getattrs, attrs=['amount', 'currency'])
        },
        'predicate': SortedList(key=attrgetter('priority'))
    },
    'compat': {
        'classname': {
            'datetime': methodcaller('isoformat'),
            'date': methodcaller('isoformat'),
            'time': methodcaller('isoformat'),
            'set': list,
            'ndarray': np_to_list,
            'float16': generic_to_item,
            'float32': generic_to_item,
            'frozenset': list,
            'complex': partial(getattrs, attrs=['real', 'imag']),
            'Fraction': partial(getattrs, attrs=['numerator', 'denominator']),
            'UUID': str,
            'Currency': str,
            'Money': str,
        },
        'predicate': SortedList(key=attrgetter('priority'))
    }
}


# all decode handlers are for EXACT decoding BY CLASSNAME
_decode_handlers = {
    'datetime': parse_datetime,
    'date': lambda v: parse_datetime(v).date(),
    'time': lambda v: parse_datetime(v).timetz(),
    'timedelta': kwargified(timedelta),
    'tuple': tuple,
    'set': set,
    'ndarray': np.asarray,
    'float16': np.float16,
    'float32': np.float32,
    'frozenset': frozenset,
    'complex': kwargified(complex),
    'Decimal': Decimal,
    'Fraction': kwargified(Fraction),
    'UUID': kwargified(uuid.UUID)
}


@encoder('namedtuple', lambda obj: isinstance(obj, tuple) and hasattr(obj, '_fields'))
def _dump_namedtuple(obj):
    return {"name": type(obj).__name__,
            "fields": list(obj._fields),
            "values": list(obj)}


@decoder('namedtuple')
def _load_namedtuple(val):
    cls = namedtuple(val['name'], val['fields'])
    return cls(*val['values'])


@encoder('timedelta', exact=False)
def _timedelta_total_seconds(td):
    # timedelta.total_seconds() is only available since python 2.7
    return (td.microseconds + (td.seconds + td.days * 24 * 3600.0) * 10**6) / 10**6


@encoder('Currency')
def _dump_currency(obj):
    """Serialize standard (ISO-defined) currencies to currency code only,
    and non-standard (user-added) currencies in full.
    """
    from moneyed import get_currency, CurrencyDoesNotExist
    try:
        get_currency(obj.code)
        return obj.code
    except CurrencyDoesNotExist:
        return getattrs(obj, ['code', 'numeric', 'name', 'countries'])


@decoder('Currency')
def _load_currency(val):
    """Deserialize string values as standard currencies, but
    manually define fully-defined currencies (with code/name/numeric/countries).
    """
    from moneyed import get_currency
    try:
        return get_currency(code=val)
    except:
        return Currency(**val)


@decoder('Money')
def _load_money(val):
    # wrap with function to delay Currency/Money
    # parsing if not installed (and not needed)
    return Money(**val)
