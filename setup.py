# Copyright (c) Alibaba, Inc. and its affiliates.
# !/usr/bin/env python
import os
import shutil
import subprocess
from setuptools import find_packages, setup

from modelscope.utils.ast_utils import generate_ast_template
from modelscope.utils.constant import Fields


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


version_file = 'modelscope/version.py'


def get_git_hash():

    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
    except OSError:
        sha = 'unknown'

    return sha


def get_hash():
    assert os.path.exists('.git'), '.git directory does not exist'
    sha = get_git_hash()[:7]
    return sha


def get_version():
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def parse_requirements(fname='requirements.txt', with_version=True):
    """
    Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """
        Parse information from a line in a requirements text file
        """
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            relative_base = os.path.dirname(fname)
            absolute_target = os.path.join(relative_base, target)
            for info in parse_require_file(absolute_target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith('http'):
                    print('skip http requirements %s' % line)
                    continue
                if line and not line.startswith('#') and not line.startswith(
                        '--'):
                    for info in parse_line(line):
                        yield info
                elif line and line.startswith('--find-links'):
                    eles = line.split()
                    for e in eles:
                        e = e.strip()
                        if 'http' in e:
                            info = dict(dependency_links=e)
                            yield info

    def gen_packages_items():
        items = []
        deps_link = []
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                if 'dependency_links' not in info:
                    parts = [info['package']]
                    if with_version and 'version' in info:
                        parts.extend(info['version'])
                    if not sys.version.startswith('3.4'):
                        # apparently package_deps are broken in 3.4
                        platform_deps = info.get('platform_deps')
                        if platform_deps is not None:
                            parts.append(';' + platform_deps)
                    item = ''.join(parts)
                    items.append(item)
                else:
                    deps_link.append(info['dependency_links'])
        return items, deps_link

    return gen_packages_items()


def pack_resource():
    # pack resource such as configs and tools
    root_dir = 'package/'
    if os.path.isdir(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir)

    proj_dir = root_dir + 'modelscope/'
    shutil.copytree('./modelscope', proj_dir)
    shutil.copytree('./configs', proj_dir + 'configs')
    shutil.copytree('./requirements', 'package/requirements')
    shutil.copy('./requirements.txt', 'package/requirements.txt')
    shutil.copy('./MANIFEST.in', 'package/MANIFEST.in')
    shutil.copy('./README.md', 'package/README.md')


if __name__ == '__main__':
    # write_version_py()
    generate_ast_template()
    pack_resource()
    os.chdir('package')
    install_requires, deps_link = parse_requirements('requirements.txt')
    extra_requires = {}
    all_requires = []
    for field in dir(Fields):
        if field.startswith('_'):
            continue
        field = getattr(Fields, field)
        extra_requires[field], _ = parse_requirements(
            f'requirements/{field}.txt')

        # skip audio requirements due to its hard dependency which
        # result in mac/windows compatibility problems
        if field != Fields.audio:
            all_requires.append(extra_requires[field])
    for subfiled in ['asr', 'kws', 'signal', 'tts']:
        filed_name = f'audio_{subfiled}'
        extra_requires[filed_name], _ = parse_requirements(
            f'requirements/audio/{filed_name}.txt')
    extra_requires['all'] = all_requires

    setup(
        name='modelscope',
        version=get_version(),
        description='',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='Alibaba ModelScope team',
        author_email='modelscope@list.alibaba-inc.com',
        keywords='python,nlp,science,cv,speech,multi-modal',
        url='https://github.com/modelscope/modelscope',
        packages=find_packages(exclude=('configs', 'demo')),
        include_package_data=True,
        package_data={
            '': ['*.h', '*.cpp', '*.cu'],
        },
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
        ],
        license='Apache License 2.0',
        tests_require=parse_requirements('requirements/tests.txt'),
        install_requires=install_requires,
        extras_require=extra_requires,
        entry_points={
            'console_scripts': ['modelscope=modelscope.cli.cli:run_cmd']
        },
        dependency_links=deps_link,
        zip_safe=False)
