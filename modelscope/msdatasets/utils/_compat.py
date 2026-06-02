# isort: skip_file
# yapf: disable
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Compatibility shims for datasets>=4.0 script-based dataset loading.

Script-based dataset loading was removed in datasets 4.0. This module
provides minimal re-implementations of the necessary helpers so that
ModelScope can still load datasets that ship a custom builder .py script.

When running with datasets<4.0 the real implementations are simply
re-exported from datasets.load / datasets.utils.py_utils.
"""
import importlib
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from datasets import DownloadMode, config

# ---------------------------------------------------------------------------
# Try importing script-loading APIs from datasets<4.0
# ---------------------------------------------------------------------------
try:
    from datasets.load import (
        HubDatasetModuleFactoryWithScript,
        LocalDatasetModuleFactoryWithScript,
        resolve_trust_remote_code,
        _get_importable_file_path,
        _create_importable_file,
        _load_importable_file,
        init_dynamic_modules,
        files_to_hash,
    )
    from datasets.utils.py_utils import get_imports

    _HAS_SCRIPT_LOADING = True
except ImportError:
    _HAS_SCRIPT_LOADING = False
    HubDatasetModuleFactoryWithScript = None  # type: ignore[assignment,misc]
    LocalDatasetModuleFactoryWithScript = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Compat implementations (only defined when datasets>=4.0)
# ---------------------------------------------------------------------------
if not _HAS_SCRIPT_LOADING:
    import filecmp
    import hashlib  # noqa: F811 – only imported in this branch
    import json as _json
    import re
    import shutil
    from urllib.parse import urlparse

    from datasets.packaged_modules import _hash_python_lines
    from datasets.utils.file_utils import url_or_path_join
    from datasets.utils.hub import hf_dataset_url  # noqa: F401
    from filelock import FileLock

    def _compat_get_imports(
            file_path: str) -> List[Tuple[str, str, str, Optional[str]]]:
        """Parse a dataset script for import statements (ported from datasets<4.0)."""
        with open(file_path, encoding='utf-8') as f:
            lines = f.readlines()

        imports: List[Tuple[str, str, str, Optional[str]]] = []
        is_in_docstring = False
        for line in lines:
            docstr_start_match = re.findall(r'[\s\S]*?"""[\s\S]*?', line)
            if len(docstr_start_match) == 1:
                is_in_docstring = not is_in_docstring
            if is_in_docstring:
                continue
            match = re.match(
                r'^import\s+(\.?)([^\s\.]+)[^#\r\n]*(?:#\s+From:\s+)?([^\r\n]*)',
                line,
                flags=re.MULTILINE)
            if match is None:
                match = re.match(
                    r'^from\s+(\.?)([^\s\.]+)(?:[^\s]*)\s+import\s+[^#\r\n]*(?:#\s+From:\s+)?([^\r\n]*)',
                    line,
                    flags=re.MULTILINE)
            if match is None:
                continue
            if match.group(1):
                if any(imp[1] == match.group(2) for imp in imports):
                    continue
                if match.group(3):
                    url_path = match.group(3)
                    url_path, sub_directory = _compat_convert_github_url(
                        url_path)
                    imports.append(
                        ('external', match.group(2), url_path, sub_directory))
                elif match.group(2):
                    imports.append(
                        ('internal', match.group(2), match.group(2), None))
            else:
                if match.group(3):
                    imports.append(
                        ('library', match.group(2), match.group(3), None))
                else:
                    imports.append(
                        ('library', match.group(2), match.group(2), None))
        return imports

    def _compat_convert_github_url(url_path: str) -> Tuple[str, Optional[str]]:
        parsed = urlparse(url_path)
        sub_directory = None
        if parsed.scheme in ('http', 'https',
                             's3') and parsed.netloc == 'github.com':
            if 'blob' in url_path:
                if not url_path.endswith('.py'):
                    raise ValueError(
                        f'External import from github at {url_path} should point to a .py file'
                    )
                url_path = url_path.replace('blob', 'raw')
            else:
                github_path = parsed.path[1:]
                repo_info, branch = (
                    github_path.split('/tree/') if '/tree/' in github_path else
                    (github_path, 'master'))
                repo_owner, repo_name = repo_info.split('/')
                url_path = f'https://github.com/{repo_owner}/{repo_name}/archive/{branch}.zip'
                sub_directory = f'{repo_name}-{branch}'
        return url_path, sub_directory

    # -- dynamic module management ----------------------------------------

    def _compat_init_dynamic_modules(
        name: str = config.MODULE_NAME_FOR_DYNAMIC_MODULES,
        hf_modules_cache=None,
    ) -> str:
        hf_modules_cache = str(hf_modules_cache or config.HF_MODULES_CACHE)
        if hf_modules_cache not in sys.path:
            sys.path.append(hf_modules_cache)
        os.makedirs(hf_modules_cache, exist_ok=True)
        init_path = os.path.join(hf_modules_cache, '__init__.py')
        if not os.path.exists(init_path):
            with open(init_path, 'w'):
                pass
            importlib.invalidate_caches()
        dynamic_modules_path = os.path.join(hf_modules_cache, name)
        os.makedirs(dynamic_modules_path, exist_ok=True)
        init_path2 = os.path.join(dynamic_modules_path, '__init__.py')
        if not os.path.exists(init_path2):
            with open(init_path2, 'w'):
                pass
        return dynamic_modules_path

    def _compat_files_to_hash(file_paths) -> str:
        to_use_files: list = []
        for fp in file_paths:
            if os.path.isdir(fp):
                to_use_files.extend(list(Path(fp).rglob('*.[pP][yY]')))
            else:
                to_use_files.append(fp)
        lines: list = []
        for fp in to_use_files:
            with open(fp, encoding='utf-8') as f:
                lines.extend(f.readlines())
        return _hash_python_lines(lines)

    # -- importable file management ---------------------------------------

    def _compat_get_importable_file_path(
        dynamic_modules_path: str,
        module_namespace: str,
        subdirectory_name: str,
        name: str,
    ) -> str:
        importable_dir = os.path.join(dynamic_modules_path, module_namespace,
                                      name.replace('/', '--'))
        return os.path.join(importable_dir, subdirectory_name,
                            name.split('/')[-1] + '.py')

    def _compat_copy_script_and_resources(
        name: str,
        importable_directory_path: str,
        subdirectory_name: str,
        original_local_path: str,
        local_imports: List[Tuple[str, str]],
        additional_files: List[Tuple[str, str]],
        download_mode,
    ) -> str:
        importable_subdirectory = os.path.join(importable_directory_path,
                                               subdirectory_name)
        importable_file = os.path.join(importable_subdirectory, name + '.py')
        lock_path = importable_directory_path + '.lock'
        with FileLock(lock_path):
            if download_mode == DownloadMode.FORCE_REDOWNLOAD and os.path.exists(
                    importable_directory_path):
                shutil.rmtree(importable_directory_path)
            os.makedirs(importable_directory_path, exist_ok=True)
            init_fp = os.path.join(importable_directory_path, '__init__.py')
            if not os.path.exists(init_fp):
                with open(init_fp, 'w'):
                    pass
            os.makedirs(importable_subdirectory, exist_ok=True)
            init_fp2 = os.path.join(importable_subdirectory, '__init__.py')
            if not os.path.exists(init_fp2):
                with open(init_fp2, 'w'):
                    pass
            if not os.path.exists(importable_file):
                shutil.copyfile(original_local_path, importable_file)
                meta_path = os.path.splitext(importable_file)[0] + '.json'
                if not os.path.exists(meta_path):
                    meta = {
                        'original file path': original_local_path,
                        'local file path': importable_file
                    }
                    with open(meta_path, 'w', encoding='utf-8') as mf:
                        _json.dump(meta, mf)
            for imp_name, imp_path in local_imports:
                if os.path.isfile(imp_path):
                    dest = os.path.join(importable_subdirectory,
                                        imp_name + '.py')
                    if not os.path.exists(dest):
                        shutil.copyfile(imp_path, dest)
                elif os.path.isdir(imp_path):
                    dest = os.path.join(importable_subdirectory, imp_name)
                    if not os.path.exists(dest):
                        shutil.copytree(imp_path, dest)
                else:
                    raise ImportError(f'Error with local import at {imp_path}')
            for file_name, original_path in additional_files:
                dest_path = os.path.join(importable_subdirectory, file_name)
                if not os.path.exists(dest_path) or not filecmp.cmp(
                        original_path, dest_path):
                    shutil.copyfile(original_path, dest_path)
        return importable_file

    def _compat_create_importable_file(
        local_path: str,
        local_imports: List[Tuple[str, str]],
        additional_files: List[Tuple[str, str]],
        dynamic_modules_path: str,
        module_namespace: str,
        subdirectory_name: str,
        name: str,
        download_mode,
    ) -> None:
        importable_dir = os.path.join(dynamic_modules_path, module_namespace,
                                      name.replace('/', '--'))
        Path(importable_dir).mkdir(parents=True, exist_ok=True)
        (Path(importable_dir).parent / '__init__.py').touch(exist_ok=True)
        _compat_copy_script_and_resources(
            name=name.split('/')[-1],
            importable_directory_path=importable_dir,
            subdirectory_name=subdirectory_name,
            original_local_path=local_path,
            local_imports=local_imports,
            additional_files=additional_files,
            download_mode=download_mode,
        )

    def _compat_load_importable_file(
        dynamic_modules_path: str,
        module_namespace: str,
        subdirectory_name: str,
        name: str,
    ) -> Tuple[str, str]:
        module_path = '.'.join([
            os.path.basename(dynamic_modules_path),
            module_namespace,
            name.replace('/', '--'),
            subdirectory_name,
            name.split('/')[-1],
        ])
        return module_path, subdirectory_name

    # -- trust handling ---------------------------------------------------

    def _compat_resolve_trust_remote_code(trust_remote_code, repo_id: str):
        if trust_remote_code is None:
            raise ValueError(
                f'The repository for {repo_id} contains custom code which must be '
                f'executed to correctly load the dataset. You can inspect the repository '
                f'content at the Hub.\nPlease pass the argument `trust_remote_code=True` '
                f'to allow custom code to be run.')
        return trust_remote_code

    # -- Assign compat functions to canonical names -----------------------
    get_imports = _compat_get_imports  # noqa: F811
    init_dynamic_modules = _compat_init_dynamic_modules  # noqa: F811
    files_to_hash = _compat_files_to_hash  # noqa: F811
    resolve_trust_remote_code = _compat_resolve_trust_remote_code  # noqa: F811
    _get_importable_file_path = _compat_get_importable_file_path  # noqa: F811
    _create_importable_file = _compat_create_importable_file  # noqa: F811
    _load_importable_file = _compat_load_importable_file  # noqa: F811
