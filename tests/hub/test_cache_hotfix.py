"""Test legacy cache helpers and SDK cleanup entry points."""

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest
from modelscope_hub import config
from modelscope_hub.errors import CacheError

from modelscope import snapshot_download
from modelscope.cli.clearcache import ClearCacheCMD
from modelscope.hub.cache_manager import (get_cache_root,
                                          resolve_cached_snapshot)
from modelscope.hub.utils.caching import FileSystemCache, ModelFileSystemCache
from modelscope.utils.file_utils import (get_hub_cache_root,
                                         get_model_cache_root,
                                         get_modelscope_cache_dir)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path / 'cache'
    monkeypatch.setenv('MODELSCOPE_CACHE', str(root))
    monkeypatch.setenv('MODELSCOPE_HOME', str(tmp_path / 'auth'))
    monkeypatch.setattr(config, '_default_config', None)
    path = root / 'models/org--model/snapshots/master'
    path.mkdir(parents=True)
    (path / 'config.json').write_text('{}')
    return root, path


def test_helpers_keep_legacy_contract_and_expose_actual_snapshot(cache):
    root, path = cache
    assert get_modelscope_cache_dir() == str(root)
    assert get_model_cache_root() == str(root / 'models')
    assert get_hub_cache_root() == str(root) and get_cache_root() == root
    assert resolve_cached_snapshot('org/model') == path
    with patch('requests.Session.send', side_effect=AssertionError('不得联网')):
        assert snapshot_download('org/model', None, root, None,
                                 True) == str(path)
    assert 'legacy SDK' in get_model_cache_root.__doc__
    assert 'resolve_cached_snapshot' in get_model_cache_root.__doc__


@pytest.mark.parametrize('model', ['', '..', '../..', 'org/..', '/tmp/x'])
def test_legacy_command_rejects_bad_ids_before_confirmation(cache, model):
    root, path = cache
    with patch('builtins.input', side_effect=AssertionError('错误参数不得询问确认')):
        with pytest.raises(CacheError):
            ClearCacheCMD(Namespace(model=model, dataset=None)).execute()
    assert path.exists() and root.exists()


def test_legacy_command_clears_only_repository(cache):
    root, path = cache
    sentinel = root / 'hub/ast_indexer/sentinel'
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text('keep')
    with patch('builtins.input', return_value='Y'):
        ClearCacheCMD(Namespace(model='org/model', dataset=None)).execute()
    assert not path.exists() and sentinel.exists() and root.exists()


def test_legacy_remove_never_follows_home_link(cache, tmp_path):
    root, path = cache
    home = tmp_path / 'fake_home'
    home.mkdir()
    (home / 'sentinel').write_text('keep')
    link = root / 'models/home_link'
    link.symlink_to(home, target_is_directory=True)
    cmd = ClearCacheCMD(Namespace(model=None, dataset=None))
    with pytest.raises(CacheError):
        cmd._remove_directory(str(link))
    assert (home / 'sentinel').exists() and path.exists()


def test_file_system_cache_cannot_clear_home(cache, tmp_path, monkeypatch):
    home = tmp_path / 'fake_home'
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: home))
    target = FileSystemCache(str(home))
    with pytest.raises(CacheError):
        target.clear_cache()
    assert home.exists()


def test_model_cache_clear_preserves_explicit_root(cache, tmp_path):
    root = tmp_path / 'legacy'
    target = ModelFileSystemCache(str(root), 'org', 'model')
    target.clear_cache()
    assert root.is_dir() and not (root / 'org/model').exists()


def test_model_cache_clear_supports_relative_root(cache, tmp_path,
                                                  monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = ModelFileSystemCache('relative', 'org', 'model')
    target.clear_cache()
    assert (tmp_path / 'relative').is_dir()
    assert not (tmp_path / 'relative/org/model').exists()


def test_model_cache_rejects_owner_path_traversal(tmp_path):
    with pytest.raises(CacheError):
        ModelFileSystemCache(str(tmp_path / 'legacy'), '..', 'model')


def test_vllm_lora_exception_import_is_compatible():
    from modelscope.hub.snapshot_download import InvalidParameter
    from modelscope_hub.errors import InvalidParameter as Expected
    assert InvalidParameter is Expected


@pytest.mark.parametrize('revision', [None, '', 'master'])
def test_offline_model_file_skips_revision_lookup(cache, revision):
    from modelscope.hub.file_download import model_file_download
    from modelscope_hub.config import HubConfig
    root, path = cache
    with (
            patch.object(
                HubConfig,
                '__post_init__',
                side_effect=AssertionError('不得初始化认证')),
            patch('requests.Session.send', side_effect=AssertionError('不得联网')),
    ):
        assert model_file_download(
            'org/model',
            'config.json',
            revision=revision,
            local_files_only=True) == str(path / 'config.json')


def test_audio_update_reuses_standard_snapshot(cache):
    from modelscope.utils.audio.audio_utils import update_local_model
    from modelscope_hub._download import DownloadManager
    root, path = cache
    with patch.object(
            DownloadManager,
            'download_repo',
            side_effect=AssertionError('不得重复整仓下载')) as download:
        update_local_model({'model': 'org/model'}, str(path), {})
    download.assert_not_called()
    assert not (root / 'hub/models/.cache').exists()


def test_audio_update_does_not_confuse_sibling_prefix(cache, tmp_path):
    from modelscope.utils.audio.audio_utils import update_local_model
    from modelscope_hub._download import DownloadManager
    root, path = cache
    external = root / 'models-external/model'
    external.mkdir(parents=True)
    with patch.object(
            DownloadManager, 'download_repo',
            return_value=external) as download:
        update_local_model({'model': 'org/model'}, str(external), {})
    download.assert_called_once()
