# Copyright (c) Alibaba, Inc. and its affiliates.
# This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
# Part of the implementation is borrowed from wimglenn/johnnydep

import copy
import filecmp
import importlib
import os
import pkgutil
import shutil
import sys
from contextlib import contextmanager
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Iterable, List, Optional, Set, Union

import json
import pkg_resources

from modelscope import snapshot_download
from modelscope.fileio.file import LocalStorage
from modelscope.utils.ast_utils import FilesAstScanning
from modelscope.utils.constant import DEFAULT_MODEL_REVISION
from modelscope.utils.file_utils import get_modelscope_cache_dir
from modelscope.utils.logger import get_logger

logger = get_logger()
storage = LocalStorage()

MODELSCOPE_FILE_DIR = get_modelscope_cache_dir()
MODELSCOPE_DYNAMIC_MODULE = 'modelscope_modules'
BASE_MODULE_DIR = os.path.join(MODELSCOPE_FILE_DIR, MODELSCOPE_DYNAMIC_MODULE)

PLUGINS_FILENAME = '.modelscope_plugins'
OFFICIAL_PLUGINS = [
    {
        'name': 'adaseq',
        'desc':
        'Provide hundreds of additions NERs algorithms, check: https://github.com/modelscope/AdaSeq',
        'version': '',
        'url': ''
    },
]

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


def discover_plugins(requirement_path=None) -> Iterable[str]:
    """
    Discover plugins

        Args:
        requirement_path: The file path of requirement

    """
    plugins: Set[str] = set()
    if requirement_path is None:
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
    else:
        if os.path.isfile(requirement_path):
            for plugin in discover_file_plugins(requirement_path):
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
            # TODO: include and exclude should be configurable, hard code now
            import_module_and_submodules(
                module_name,
                include={
                    'easycv.toolkit.modelscope',
                    'easycv.hooks',
                    'easycv.models',
                    'easycv.core',
                    'easycv.toolkit',
                    'easycv.predictors',
                },
                exclude={
                    'easycv.toolkit.*',
                    'easycv.*',
                })
            logger.info('Plugin %s available', module_name)
            imported_plugins.append(module_name)
        except ModuleNotFoundError as e:
            logger.error(f'Plugin {module_name} could not be loaded: {e}')

    return imported_plugins


def import_file_plugins(requirement_path=None) -> List[str]:
    """
    Imports the plugins found with `discover_plugins()`.

    Args:
        requirement_path: The file path of requirement

    """
    imported_plugins: List[str] = []

    # Workaround for a presumed Python issue where spawned processes can't find modules in the current directory.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.append(cwd)

    for module_name in discover_plugins(requirement_path):
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

    def fn_in(package_name: str, pattern_set: Set[str]) -> bool:
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
            for module_finder, name, _ in pkgutil.iter_modules(path):
                # Sometimes when you import third-party libraries that are on your path,
                # `pkgutil.walk_packages` returns those too, so we need to skip them.
                # `pkgutil.iter_modules` avoid import those package
                if path_string and module_finder.path != path_string:  # type: ignore[union-attr]
                    continue
                if name.startswith('_'):
                    # skip directly importing private subpackages
                    continue
                if name.startswith('test'):
                    # skip tests
                    continue
                subpackage = f'{package_name}.{name}'
                import_module_and_submodules(
                    subpackage, include=include, exclude=exclude)
        except SystemExit as e:
            # this case is specific for easy_cv's tools/predict.py exit
            logger.warning(
                f'{package_name} not imported: {str(e)}, but should continue')
            pass
        except Exception as e:
            logger.warning(f'{package_name} not imported: {str(e)}')
            if len(package_name.split('.')) == 1:
                raise ModuleNotFoundError('Package not installed')


def install_module_from_requirements(requirement_path, ):
    """ install module from requirements
    Args:
        requirement_path: The path of requirement file

    No returns, raise error if failed
    """

    install_list = []
    with open(requirement_path, 'r', encoding='utf-8') as f:
        requirements = f.read().splitlines()
        for req in requirements:
            if req == '':
                continue
            installed, _ = PluginsManager.check_plugin_installed(req)
            if not installed:
                install_list.append(req)

    if len(install_list) > 0:
        status_code, _, args = PluginsManager.pip_command(
            'install',
            install_list,
        )
        if status_code != 0:
            raise ImportError(
                f'Failed to install requirements from {requirement_path}')


def import_module_from_file(module_name, file_path):
    """ install module by name with file path

    Args:
        module_name: the module name need to be import
        file_path: the related file path that matched with the module name

    Returns: return the module class

    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_module_from_files(file_list, file_prefix, module_name):
    """
    Create a python module from a list of files by copying them to the destination directory.

    Args:
        file_list (List[str]): List of relative file paths to be copied.
        file_prefix (str): Path prefix for each file in file_list.
        module_name (str): Name of the module.

    Returns:
        None
    """

    def create_empty_file(file_path):
        with open(file_path, 'w') as _:
            pass

    dest_dir = os.path.join(BASE_MODULE_DIR, module_name)
    for file_path in file_list:
        file_dir = os.path.dirname(file_path)
        target_dir = os.path.join(dest_dir, file_dir)
        os.makedirs(target_dir, exist_ok=True)
        init_file = os.path.join(target_dir, '__init__.py')
        if not os.path.exists(init_file):
            create_empty_file(init_file)

        target_file = os.path.join(target_dir, os.path.basename(file_path))
        src_file = os.path.join(file_prefix, file_path)
        if not os.path.exists(target_file) or not filecmp.cmp(
                src_file, target_file):
            shutil.copyfile(src_file, target_file)

    importlib.invalidate_caches()


def import_module_from_model_dir(model_dir):
    """ import all the necessary module from a model dir

    Args:
        model_dir: model file location

     No returns, raise error if failed

    """
    from pathlib import Path
    file_scanner = FilesAstScanning()
    file_scanner.traversal_files(model_dir, include_init=True)
    file_dirs = file_scanner.file_dirs
    requirements = file_scanner.requirement_dirs

    # install the requirements firstly
    install_requirements_by_files(requirements)

    if BASE_MODULE_DIR not in sys.path:
        sys.path.append(BASE_MODULE_DIR)

    module_name = Path(model_dir).stem

    # in order to keep forward compatibility, we add module path to
    # sys.path so that submodule can be imported directly as before
    MODULE_PATH = os.path.join(BASE_MODULE_DIR, module_name)
    if MODULE_PATH not in sys.path:
        sys.path.append(MODULE_PATH)

    relative_file_dirs = [
        file.replace(model_dir.rstrip(os.sep) + os.sep, '')
        for file in file_dirs
    ]
    create_module_from_files(relative_file_dirs, model_dir, module_name)
    for file in relative_file_dirs:
        submodule = module_name + '.' + file.replace('.py', '').replace(os.sep, '.')
        importlib.import_module(submodule)


def install_requirements_by_names(plugins: List[str]):
    """ install the requirements by names

    Args:
        plugins: name of plugins (pai-easyscv, transformers)

     No returns, raise error if failed

    """
    plugins_manager = PluginsManager()
    uninstalled_plugins = []
    for plugin in plugins:
        plugin_installed, version = plugins_manager.check_plugin_installed(
            plugin)
        if not plugin_installed:
            uninstalled_plugins.append(plugin)
    status, _ = plugins_manager.install_plugins(uninstalled_plugins)
    if status != 0:
        raise EnvironmentError(
            f'The required packages {",".join(uninstalled_plugins)} are not installed.',
            f'Please run the command `modelscope plugin install {" ".join(uninstalled_plugins)}` to install them.'
        )


def install_requirements_by_files(requirements: List[str]):
    """ install the requriements by files

    Args:
        requirements: a list of files including requirements info (requirements.txt)

     No returns, raise error if failed

    """
    for requirement in requirements:
        install_module_from_requirements(requirement)


def register_plugins_repo(plugins: List[str]) -> None:
    """ Try to install and import plugins from repo"""
    if plugins is not None:
        install_requirements_by_names(plugins)
        modules = []
        for plugin in plugins:
            module_name, module_version, _ = get_modules_from_package(plugin)
            modules.extend(module_name)
        import_plugins(modules)


def register_modelhub_repo(model_dir, allow_remote=False) -> None:
    """ Try to install and import remote model from modelhub"""
    if allow_remote:
        try:
            import_module_from_model_dir(model_dir)
        except KeyError:
            logger.warning(
                'Multi component keys in the hub are registered in same file')
            pass


DEFAULT_INDEX = 'https://pypi.org/simple/'


def get_modules_from_package(package):
    """ to get the modules from an installed package

    Args:
        package: The distribution name or package name

    Returns:
        import_names: The modules that in the package distribution
        import_version: The version of those modules, should be same and identical
        package_name: The package name, if installed by whl file, the package is unknown, should be passed

    """
    from zipfile import ZipFile
    from tempfile import mkdtemp
    from subprocess import check_output, STDOUT
    from glob import glob
    import hashlib
    from urllib.parse import urlparse
    from urllib import request as urllib2
    from pip._internal.utils.packaging import get_requirement

    def urlretrieve(url, filename, data=None, auth=None):
        if auth is not None:
            # https://docs.python.org/2.7/howto/urllib2.html#id6
            password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()

            # Add the username and password.
            # If we knew the realm, we could use it instead of None.
            username, password = auth
            top_level_url = urlparse(url).netloc
            password_mgr.add_password(None, top_level_url, username, password)

            handler = urllib2.HTTPBasicAuthHandler(password_mgr)

            # create "opener" (OpenerDirector instance)
            opener = urllib2.build_opener(handler)
        else:
            opener = urllib2.build_opener()

        res = opener.open(url, data=data)

        headers = res.info()

        with open(filename, 'wb') as fp:
            fp.write(res.read())

        return filename, headers

    def compute_checksum(target, algorithm='sha256', blocksize=2**13):
        hashtype = getattr(hashlib, algorithm)
        hash_ = hashtype()
        logger.debug('computing checksum', target=target, algorithm=algorithm)
        with open(target, 'rb') as f:
            for chunk in iter(lambda: f.read(blocksize), b''):
                hash_.update(chunk)
        result = hash_.hexdigest()
        logger.debug('computed checksum', result=result)
        return result

    def _get_pip_version():
        # try to get pip version without actually importing pip
        # setuptools gets upset if you import pip before importing setuptools..
        try:
            import importlib.metadata  # Python 3.8+
            return importlib.metadata.version('pip')
        except Exception:
            pass
        import pip
        return pip.__version__

    def _download_dist(url, scratch_file, index_url, extra_index_url):
        auth = None
        if index_url:
            parsed = urlparse(index_url)
            if parsed.username and parsed.password and parsed.hostname == urlparse(
                    url).hostname:
                # handling private PyPI credentials in index_url
                auth = (parsed.username, parsed.password)
        if extra_index_url:
            parsed = urlparse(extra_index_url)
            if parsed.username and parsed.password and parsed.hostname == urlparse(
                    url).hostname:
                # handling private PyPI credentials in extra_index_url
                auth = (parsed.username, parsed.password)
        target, _headers = urlretrieve(url, scratch_file, auth=auth)
        return target, _headers

    def _get_wheel_args(index_url, env, extra_index_url):
        args = [
            sys.executable,
            '-m',
            'pip',
            'wheel',
            '-vvv',  # --verbose x3
            '--no-deps',
            '--no-cache-dir',
            '--disable-pip-version-check',
        ]
        if index_url is not None:
            args += ['--index-url', index_url]
            if index_url != DEFAULT_INDEX:
                hostname = urlparse(index_url).hostname
                if hostname:
                    args += ['--trusted-host', hostname]
        if extra_index_url is not None:
            args += [
                '--extra-index-url', extra_index_url, '--trusted-host',
                urlparse(extra_index_url).hostname
            ]
        if env is None:
            pip_version = _get_pip_version()
        else:
            pip_version = dict(env)['pip_version']
            args[0] = dict(env)['python_executable']
        pip_major, pip_minor = pip_version.split('.')[0:2]
        pip_major = int(pip_major)
        pip_minor = int(pip_minor)
        if pip_major >= 10:
            args.append('--progress-bar=off')
        if (20, 3) <= (pip_major, pip_minor) < (21, 1):
            # See https://github.com/pypa/pip/issues/9139#issuecomment-735443177
            args.append('--use-deprecated=legacy-resolver')
        return args

    def get(dist_name,
            index_url=None,
            env=None,
            extra_index_url=None,
            tmpdir=None,
            ignore_errors=False):
        args = _get_wheel_args(index_url, env, extra_index_url) + [dist_name]
        scratch_dir = mkdtemp(dir=tmpdir)
        logger.debug(
            'wheeling and dealing',
            scratch_dir=os.path.abspath(scratch_dir),
            args=' '.join(args))
        try:
            out = check_output(
                args, stderr=STDOUT, cwd=scratch_dir).decode('utf-8')
        except ChildProcessError as err:
            out = getattr(err, 'output', b'').decode('utf-8')
            logger.warning(out)
            if not ignore_errors:
                raise
        logger.debug('wheel command completed ok', dist_name=dist_name)
        links = []
        local_links = []
        lines = out.splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('Downloading from URL '):
                parts = line.split()
                link = parts[3]
                links.append(link)
            elif line.startswith('Downloading '):
                parts = line.split()
                last = parts[-1]
                if len(parts) == 3 and last.startswith('(') and last.endswith(
                        ')'):
                    link = parts[-2]
                elif len(parts) == 4 and parts[-2].startswith(
                        '(') and last.endswith(')'):
                    link = parts[-3]
                    if not urlparse(link).scheme:
                        # newest pip versions have changed to not log the full url
                        # in the download event. it is becoming more and more annoying
                        # to preserve compatibility across a wide range of pip versions
                        next_line = lines[i + 1].strip()
                        if next_line.startswith(
                                'Added ') and ' to build tracker' in next_line:
                            link = next_line.split(
                                ' to build tracker')[0].split()[-1]
                else:
                    link = last
                links.append(link)
            elif line.startswith(
                    'Source in ') and 'which satisfies requirement' in line:
                link = line.split()[-1]
                links.append(link)
            elif line.startswith('Added ') and ' from file://' in line:
                [link] = [x for x in line.split() if x.startswith('file://')]
                local_links.append(link)
        if not links:
            # prefer http scheme over file
            links += local_links
        links = list(dict.fromkeys(links))  # order-preserving dedupe
        if not links:
            logger.warning('could not find download link', out=out)
            raise Exception('failed to collect dist')
        if len(links) == 2:
            # sometimes we collect the same link, once with a url fragment/checksum and once without
            first, second = links
            if first.startswith(second):
                del links[1]
            elif second.startswith(first):
                del links[0]
        if len(links) > 1:
            logger.debug('more than 1 link collected', out=out, links=links)
            # Since PEP 517, maybe an sdist will also need to collect other distributions
            # for the build system, even with --no-deps specified. pendulum==1.4.4 is one
            # example, which uses poetry and doesn't publish any python37 wheel to PyPI.
            # However, the dist itself should still be the first one downloaded.
        link = links[0]
        whls = glob(os.path.join(os.path.abspath(scratch_dir), '*.whl'))
        try:
            [whl] = whls
        except ValueError:
            if ignore_errors:
                whl = ''
            else:
                raise
        url, _sep, checksum = link.partition('#')
        url = url.replace(
            '/%2Bf/', '/+f/'
        )  # some versions of pip did not unquote this fragment in the log
        if not checksum.startswith('md5=') and not checksum.startswith(
                'sha256='):
            # PyPI gives you the checksum in url fragment, as a convenience. But not all indices are so kind.
            algorithm = 'md5'
            if os.path.basename(whl).lower() == url.rsplit('/', 1)[-1].lower():
                target = whl
            else:
                scratch_file = os.path.join(scratch_dir, os.path.basename(url))
                target, _headers = _download_dist(url, scratch_file, index_url,
                                                  extra_index_url)
            checksum = compute_checksum(target=target, algorithm=algorithm)
            checksum = '='.join([algorithm, checksum])
        result = {'path': whl, 'url': url, 'checksum': checksum}
        return result

    def discover_import_names(whl_file):
        import re
        logger.debug('finding import names')
        zipfile = ZipFile(file=whl_file)
        namelist = zipfile.namelist()
        [top_level_fname
         ] = [x for x in namelist if x.endswith('top_level.txt')]
        [metadata_fname
         ] = [x for x in namelist if x.endswith('.dist-info/METADATA')]
        all_names = zipfile.read(top_level_fname).decode(
            'utf-8').strip().splitlines()
        metadata = zipfile.read(metadata_fname).decode('utf-8')
        public_names = [n for n in all_names if not n.startswith('_')]

        version_pattern = re.compile(r'^Version: (?P<version>.+)$',
                                     re.MULTILINE)
        name_pattern = re.compile(r'^Name: (?P<name>.+)$', re.MULTILINE)

        version_match = version_pattern.search(metadata)
        name_match = name_pattern.search(metadata)

        module_version = version_match.group('version')
        module_name = name_match.group('name')

        return public_names, module_version, module_name

    tmpdir = mkdtemp()
    if package.endswith('.whl'):
        """if user using .whl file then parse the whl to get the module name"""
        if not os.path.isfile(package):
            file_name = os.path.basename(package)
            file_path = os.path.join(tmpdir, file_name)
            whl_file, _ = _download_dist(package, file_path, None, None)
        else:
            whl_file = package
    else:
        """if user using package name then generate whl file and parse the file to get the module name by
        the discover_import_names method
        """
        req = get_requirement(package)
        package = req.name
        data = get(package, tmpdir=tmpdir)
        whl_file = data['path']
    import_names, import_version, package_name = discover_import_names(
        whl_file)
    shutil.rmtree(tmpdir)
    return import_names, import_version, package_name


class PluginsManager(object):
    """
    plugins manager class
    """

    def __init__(self,
                 cache_dir=MODELSCOPE_FILE_DIR,
                 plugins_file=PLUGINS_FILENAME):
        cache_dir = os.getenv('MODELSCOPE_CACHE', cache_dir)
        plugins_file = os.getenv('MODELSCOPE_PLUGINS_FILE', plugins_file)
        self._file_path = os.path.join(cache_dir, plugins_file)

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, value):
        self._file_path = value

    @staticmethod
    def check_plugin_installed(package):
        """ Check if the plugin is installed, and if the version is valid

        Args:
            package: the package name need to be installed

        Returns:
            if_installed: True if installed
            version: the version of installed or None if not installed

        """

        if package.split('.')[-1] == 'whl':
            # install from whl should test package name instead of module name
            _, module_version, package_name = get_modules_from_package(package)
            local_installed, version = PluginsManager._check_plugin_installed(
                package_name)
            if local_installed and module_version != version:
                return False, version
            elif not local_installed:
                return False, version
            return True, module_version
        else:
            return PluginsManager._check_plugin_installed(package)

    @staticmethod
    def _check_plugin_installed(package, verified_version=None):
        from pip._internal.utils.packaging import get_requirement, specifiers
        req = get_requirement(package)

        try:
            importlib.reload(pkg_resources)
            package_meta_info = pkg_resources.working_set.by_key[req.name]
            version = package_meta_info.version

            # To test if the package is installed
            installed = True

            # If installed, test if the version is correct
            for spec in req.specifier:
                installed_valid_version = spec.contains(version)
                if not installed_valid_version:
                    installed = False
                    break

        except KeyError:
            version = ''
            installed = False

        if installed and verified_version is not None and verified_version != version:
            return False, verified_version
        else:
            return installed, version

    @staticmethod
    def pip_command(
        command,
        command_args: List[str],
    ):
        """

        Args:
            command: install, uninstall command
            command_args: the args to be used with command, should be in list
              such as ['-r', 'requirements']

        Returns:
            status_code: The pip command status code, 0 if success, else is failed
            options: parsed option from system args by pip command
            args: the unknown args that could be parsed by pip command

        """
        from pip._internal.commands import create_command
        importlib.reload(pkg_resources)
        if command == 'install':
            command_args.append('-f')
            command_args.append(
                'https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html'
            )
        command = create_command(command)
        options, args = command.parse_args(command_args)

        status_code = command.main(command_args)

        # reload the pkg_resources in order to get the latest pkgs information
        importlib.reload(pkg_resources)

        return status_code, options, args

    def install_plugins(self,
                        install_args: List[str],
                        index_url: Optional[str] = None,
                        force_update=False) -> Any:
        """Install packages via pip
            Args:
            install_args (list): List of arguments passed to `pip install`.
            index_url (str, optional): The pypi index url.
            force_update: If force update on or off
        """

        if len(install_args) == 0:
            return 0, []

        if index_url is not None:
            install_args += ['-i', index_url]

        if force_update is not False:
            install_args += ['-f']

        status_code, options, args = PluginsManager.pip_command(
            'install',
            install_args,
        )

        if status_code == 0:
            logger.info(f'The plugins {",".join(args)} is installed')

            # TODO Add Ast index for ast update record

            # Add the plugins info to the local record
            installed_package = self.parse_args_info(args, options)
            self.update_plugins_file(installed_package)

        return status_code, install_args

    def parse_args_info(self, args: List[str], options):
        """
        parse arguments input info
        Args:
            args: the list of args from pip command output
            options: the options that parsed from system args by pip command method

        Returns:
            installed_package: generate installed package info in order to store in the file
                                the info includes: name, url and desc of the package
        """
        installed_package = []

        # the case of install with requirements
        if len(args) == 0:
            src_dir = options.src_dir
            requirements = options.requirments
            for requirement in requirements:
                package_info = {
                    'name': requirement,
                    'url': os.path.join(src_dir, requirement),
                    'desc': '',
                    'version': ''
                }

                installed_package.append(package_info)

        def get_package_info(package_name):
            from pathlib import Path
            package_info = {
                'name': package_name,
                'url': options.index_url,
                'desc': ''
            }

            # the case with git + http
            if package_name.split('.')[-1] == 'git':
                package_name = Path(package_name).stem

            plugin_installed, version = self.check_plugin_installed(
                package_name)
            if plugin_installed:
                package_info['version'] = version
                package_info['name'] = package_name
            else:
                logger.warning(
                    f'The package {package_name} is not in the lib, this might be happened'
                    f' when installing the package with git+https method, should be ignored'
                )
                package_info['version'] = ''

            return package_info

        for package in args:
            package_info = get_package_info(package)
            installed_package.append(package_info)

        return installed_package

    def uninstall_plugins(self,
                          uninstall_args: Union[str, List],
                          is_yes=False):
        """
        uninstall plugins
        Args:
            uninstall_args: args used to uninstall by pip command
            is_yes: force yes without verified

        Returns: status code, and uninstall args

        """
        if is_yes is not None:
            uninstall_args += ['-y']

        status_code, options, args = PluginsManager.pip_command(
            'uninstall',
            uninstall_args,
        )

        if status_code == 0:
            logger.info(f'The plugins {",".join(args)} is uninstalled')

            # TODO Add Ast index for ast update record

            # Add to the local record
            self.remove_plugins_from_file(args)

        return status_code, uninstall_args

    def _get_plugins_from_file(self):
        """ get plugins from file

        """
        logger.info(f'Loading plugins information from {self.file_path}')
        if os.path.exists(self.file_path):
            local_plugins_info_bytes = storage.read(self.file_path)
            local_plugins_info = json.loads(local_plugins_info_bytes)
        else:
            local_plugins_info = {}
        return local_plugins_info

    def _update_plugins(
        self,
        new_plugins_list,
        local_plugins_info,
        override=False,
    ):
        for item in new_plugins_list:
            package_name = item.pop('name')

            # update package information if existed
            if package_name in local_plugins_info and not override:
                original_item = local_plugins_info[package_name]
                from pkg_resources import parse_version
                item_version = parse_version(
                    item['version'] if item['version'] != '' else '0.0.0')
                origin_version = parse_version(
                    original_item['version']
                    if original_item['version'] != '' else '0.0.0')
                desc = item['desc']
                if original_item['desc'] != '' and desc == '':
                    desc = original_item['desc']
                item = item if item_version > origin_version else original_item
                item['desc'] = desc

            # Double-check if the item is installed with the version number
            if item['version'] == '':
                plugin_installed, version = self.check_plugin_installed(
                    package_name)
                item['version'] = version

            local_plugins_info[package_name] = item

        return local_plugins_info

    def _print_plugins_info(self, local_plugins_info):
        print('{:<15} |{:<10}  |{:<100}'.format('NAME', 'VERSION',
                                                'DESCRIPTION'))
        print('')
        for k, v in local_plugins_info.items():
            print('{:<15} |{:<10} |{:<100}'.format(k, v['version'], v['desc']))

    def list_plugins(
        self,
        show_all=False,
    ):
        """

        Args:
            show_all: show installed and official supported if True, else only those installed

        Returns:
            local_plugins_info: show the list of plugins info

        """
        local_plugins_info = self._get_plugins_from_file()

        # update plugins with default

        local_official_plugins = copy.deepcopy(OFFICIAL_PLUGINS)
        local_plugins_info = self._update_plugins(local_official_plugins,
                                                  local_plugins_info)

        if show_all is True:
            self._print_plugins_info(local_plugins_info)
            return local_plugins_info

        # Consider those package with version is installed
        not_installed_list = []
        for item in local_plugins_info:
            if local_plugins_info[item]['version'] == '':
                not_installed_list.append(item)

        for item in not_installed_list:
            local_plugins_info.pop(item)

        self._print_plugins_info(local_plugins_info)
        return local_plugins_info

    def update_plugins_file(
        self,
        plugins_list,
        override=False,
    ):
        """update the plugins file in order to maintain the latest plugins information

        Args:
            plugins_list: The plugins list contain the information of plugins
                name, version, introduction, install url and the status of delete or update
            override: Override the file by the list if True, else only update.

        Returns:
            local_plugins_info_json: the json version of updated plugins info

        """
        local_plugins_info = self._get_plugins_from_file()

        # local_plugins_info is empty if first time loading, should add OFFICIAL_PLUGINS information
        if local_plugins_info == {}:
            plugins_list.extend(copy.deepcopy(OFFICIAL_PLUGINS))

        local_plugins_info = self._update_plugins(plugins_list,
                                                  local_plugins_info, override)

        local_plugins_info_json = json.dumps(local_plugins_info)
        storage.write(local_plugins_info_json.encode(), self.file_path)

        return local_plugins_info_json

    def remove_plugins_from_file(
        self,
        package_names: Union[str, list],
    ):
        """remove the plugins from file
        Args:
            package_names:  package name

        Returns:
            local_plugins_info_json: the json version of updated plugins info

        """
        local_plugins_info = self._get_plugins_from_file()

        if type(package_names) is str:
            package_names = list(package_names)

        for item in package_names:
            if item in local_plugins_info:
                local_plugins_info.pop(item)

        local_plugins_info_json = json.dumps(local_plugins_info)
        storage.write(local_plugins_info_json.encode(), self.file_path)

        return local_plugins_info_json


class EnvsManager(object):
    name = 'envs'

    def __init__(self,
                 model_id,
                 model_revision=DEFAULT_MODEL_REVISION,
                 cache_dir=MODELSCOPE_FILE_DIR):
        """

        Args:
            model_id:  id of the model, not dir
            model_revision: revision of the model, default as master
            cache_dir: the system modelscope cache dir
        """
        cache_dir = os.getenv('MODELSCOPE_CACHE', cache_dir)
        self.env_dir = os.path.join(cache_dir, EnvsManager.name, model_id)
        model_dir = snapshot_download(model_id, revision=model_revision)
        from modelscope.utils.hub import read_config
        cfg = read_config(model_dir)
        self.plugins = cfg.get('plugins', [])
        self.allow_remote = cfg.get('allow_remote', False)
        import venv
        self.env_builder = venv.EnvBuilder(
            system_site_packages=True,
            clear=False,
            symlinks=True,
            with_pip=False)

    def get_env_dir(self):
        return self.env_dir

    def get_activate_dir(self):
        return os.path.join(self.env_dir, 'bin', 'activate')

    def check_if_need_env(self):
        if len(self.plugins) or self.allow_remote:
            return True
        else:
            return False

    def create_env(self):
        if not os.path.exists(self.env_dir):
            os.makedirs(self.env_dir)
        try:
            self.env_builder.create(self.env_dir)
        except Exception as e:
            self.clean_env()
            raise EnvironmentError(
                f'Failed to create virtual env at {self.env_dir} with error: {e}'
            )

    def clean_env(self):
        if os.path.exists(self.env_dir):
            self.env_builder.clear_directory(self.env_dir)

    @staticmethod
    def run_process(cmd):
        import subprocess
        status, result = subprocess.getstatusoutput(cmd)
        logger.debug('The status and the results are: {}, {}'.format(
            status, result))
        if status != 0:
            raise Exception(
                'running the cmd: {} failed, with message: {}'.format(
                    cmd, result))
        return result


if __name__ == '__main__':
    install_requirements_by_files(['adaseq'])
    import_name, import_version, package_name = get_modules_from_package(
        'pai-easycv')
