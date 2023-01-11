# Copyright (c) Alibaba, Inc. and its affiliates.
# This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
import importlib
import os
import pkgutil
import sys
from contextlib import contextmanager
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, List, Optional, Set

from modelscope.utils.logger import get_logger

logger = get_logger()

LOCAL_PLUGINS_FILENAME = '.modelscope_plugins'
GLOBAL_PLUGINS_FILENAME = os.path.join(Path.home(), '.modelscope', 'plugins')
DEFAULT_PLUGINS = []


@contextmanager
def pushd(new_dir: str, verbose: bool = False):
    """
    Changes the current directory to the given path and prepends it to `sys.path`.
    This method is intended to use with `with`, so after its usage, the current
    directory will be set to the previous value.
    """
    previous_dir = os.getcwd()
    if verbose:
        logger.info(f'Changing directory to {new_dir}')  # type: ignore
    os.chdir(new_dir)
    try:
        yield
    finally:
        if verbose:
            logger.info(f'Changing directory back to {previous_dir}')
        os.chdir(previous_dir)


@contextmanager
def push_python_path(path: str):
    """
    Prepends the given path to `sys.path`.
    This method is intended to use with `with`, so after its usage, its value
    will be removed from `sys.path`.
    """
    path = Path(path).resolve()
    path = str(path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path.remove(path)


def discover_file_plugins(
        filename: str = LOCAL_PLUGINS_FILENAME) -> Iterable[str]:
    """
    Discover plugins from file
    """
    with open(filename) as f:
        for module_name in f:
            module_name = module_name.strip()
            if module_name:
                yield module_name


def discover_plugins() -> Iterable[str]:
    """
    Discover plugins
    """
    plugins: Set[str] = set()
    if os.path.isfile(LOCAL_PLUGINS_FILENAME):
        with push_python_path('.'):
            for plugin in discover_file_plugins(LOCAL_PLUGINS_FILENAME):
                if plugin in plugins:
                    continue
                yield plugin
                plugins.add(plugin)
    if os.path.isfile(GLOBAL_PLUGINS_FILENAME):
        for plugin in discover_file_plugins(GLOBAL_PLUGINS_FILENAME):
            if plugin in plugins:
                continue
            yield plugin
            plugins.add(plugin)


def import_all_plugins(plugins: List[str] = None) -> List[str]:
    """
    Imports default plugins, input plugins and file discovered plugins.
    """
    import_module_and_submodules(
        'modelscope',
        include={
            'modelscope.metrics.builder',
            'modelscope.models.builder',
            'modelscope.pipelines.builder',
            'modelscope.preprocessors.builder',
            'modelscope.trainers.builder',
        },
        exclude={
            'modelscope.metrics.*',
            'modelscope.models.*',
            'modelscope.pipelines.*',
            'modelscope.preprocessors.*',
            'modelscope.trainers.*',
            'modelscope.msdatasets',
            'modelscope.utils',
            'modelscope.exporters',
        })

    imported_plugins: List[str] = []

    imported_plugins.extend(import_plugins(DEFAULT_PLUGINS))
    imported_plugins.extend(import_plugins(plugins))
    imported_plugins.extend(import_file_plugins())

    return imported_plugins


def import_plugins(plugins: List[str] = None) -> List[str]:
    """
    Imports the plugins listed in the arguments.
    """
    imported_plugins: List[str] = []
    if plugins is None or len(plugins) == 0:
        return imported_plugins

    # Workaround for a presumed Python issue where spawned processes can't find modules in the current directory.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.append(cwd)

    for module_name in plugins:
        try:
            import_module_and_submodules(module_name)
            logger.info('Plugin %s available', module_name)
            imported_plugins.append(module_name)
        except ModuleNotFoundError as e:
            logger.error(f'Plugin {module_name} could not be loaded: {e}')

    return imported_plugins


def import_file_plugins() -> List[str]:
    """
    Imports the plugins found with `discover_plugins()`.
    """
    imported_plugins: List[str] = []

    # Workaround for a presumed Python issue where spawned processes can't find modules in the current directory.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.append(cwd)

    for module_name in discover_plugins():
        try:
            importlib.import_module(module_name)
            logger.info('Plugin %s available', module_name)
            imported_plugins.append(module_name)
        except ModuleNotFoundError as e:
            logger.error(f'Plugin {module_name} could not be loaded: {e}')

    return imported_plugins


def import_module_and_submodules(package_name: str,
                                 include: Optional[Set[str]] = None,
                                 exclude: Optional[Set[str]] = None) -> None:
    """
    Import all public submodules under the given package.
    """
    # take care of None
    include = include if include else set()
    exclude = exclude if exclude else set()

    def fn_in(packge_name: str, pattern_set: Set[str]) -> bool:
        for pattern in pattern_set:
            if fnmatch(package_name, pattern):
                return True
        return False

    if not fn_in(package_name, include) and fn_in(package_name, exclude):
        return

    importlib.invalidate_caches()

    # For some reason, python doesn't always add this by default to your path, but you pretty much
    # always want it when using `--include-package`.  And if it's already there, adding it again at
    # the end won't hurt anything.
    with push_python_path('.'):
        # Import at top level
        try:
            module = importlib.import_module(package_name)
            path = getattr(module, '__path__', [])
            path_string = '' if not path else path[0]

            # walk_packages only finds immediate children, so need to recurse.
            for module_finder, name, _ in pkgutil.walk_packages(path):
                # Sometimes when you import third-party libraries that are on your path,
                # `pkgutil.walk_packages` returns those too, so we need to skip them.
                if path_string and module_finder.path != path_string:  # type: ignore[union-attr]
                    continue
                if name.startswith('_'):
                    # skip directly importing private subpackages
                    continue
                if name.startswith('test'):
                    # skip tests
                    continue
                subpackage = f'{package_name}.{name}'
                import_module_and_submodules(subpackage, exclude=exclude)
        except Exception as e:
            logger.warning(f'{package_name} not imported: {str(e)}')
            if len(package_name.split('.')) == 1:
                raise ModuleNotFoundError('Package not installed')
