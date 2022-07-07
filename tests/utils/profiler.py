import importlib
import sys
from functools import wraps
from typing import Any, Callable, Dict, Tuple, Type


def reraise(tp, value, tb):
    try:
        if value is None:
            value = tp()
        if value.__traceback__ is not tb:
            raise value.with_traceback(tb)
        raise value
    finally:
        value = None
        tb = None


class Profiler:

    def __init__(self) -> None:
        import cProfile
        self.pr = cProfile.Profile()

    def __enter__(self):
        self.pr.enable()

    def __exit__(self, tp, exc, tb):
        self.pr.disable()
        if tp is not None:
            reraise(tp, exc, tb)

        import pstats
        ps = pstats.Stats(self.pr, stream=sys.stderr).sort_stats('tottime')
        ps.print_stats(20)


def wrapper(tp: Type[Profiler]) -> Callable[[], Callable[..., Any]]:

    def _inner(func: Callable[..., Any]) -> Callable[..., Any]:

        @wraps(func)
        def executor(*args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
            with tp():
                return func(*args, **kwargs)

        return executor

    return _inner


PIPELINE_BASE_MODULE = 'modelscope.pipelines.base'
PIPELINE_BASE_CLASS = 'Pipeline'


def enable():
    base = importlib.import_module(PIPELINE_BASE_MODULE)
    Pipeline = getattr(base, PIPELINE_BASE_CLASS)
    Pipeline.__call__ = wrapper(Profiler)(Pipeline.__call__)
