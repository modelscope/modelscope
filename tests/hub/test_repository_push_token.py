# Copyright (c) Alibaba, Inc. and its affiliates.
"""Delegation tests for ``Repository.push`` / ``DatasetRepository.push``.

Both must hand the git wrapper a remote URL with no token embedded in it: the
URL read back from an existing checkout still carries the token used at clone
time, and ``config_git_token``/``_inject_token`` deliberately leave an
already-tokened URL alone, so an embedded token silently wins over the
``git_token`` the caller just supplied.

These tests replace the git wrapper with a mock, so they need neither a
checkout nor network access.
"""
import warnings
from unittest import mock

from modelscope.hub.repository import DatasetRepository, Repository

STALE_URL = 'https://oauth2:STALE_TOKEN@www.modelscope.cn/models/acme/demo.git'
CLEAN_URL = 'https://www.modelscope.cn/models/acme/demo.git'
FRESH_TOKEN = 'fresh-token'


def _wrapper():
    """A git wrapper whose remote URL still carries the clone-time token."""
    wrapper = mock.MagicMock()
    wrapper.get_repo_remote_url.return_value = STALE_URL
    wrapper.remove_token_from_url.side_effect = (
        lambda url: url.replace('oauth2:STALE_TOKEN@', ''))
    return wrapper


def _repository(cls, wrapper):
    repo = cls.__new__(cls)
    repo.git_wrapper = wrapper
    repo.git_token = FRESH_TOKEN
    repo.model_dir = repo.repo_work_dir = '/tmp/acme-demo'
    repo.model_base_dir = repo.repo_base_dir = '/tmp'
    repo.model_repo_name = repo.repo_name = 'acme/demo'
    repo.repo_url = CLEAN_URL
    return repo


def _push(repo, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        if isinstance(repo, DatasetRepository):
            repo.push(commit_message='msg')
        else:
            repo.push(commit_message='msg')


def test_repository_push_sends_a_url_without_the_stale_token():
    wrapper = _wrapper()
    _push(_repository(Repository, wrapper))

    assert wrapper.push.call_count == 1
    sent = wrapper.push.call_args.kwargs['url']
    assert 'STALE_TOKEN' not in sent
    assert sent == CLEAN_URL
    # The caller's token is the only credential left.
    assert wrapper.push.call_args.kwargs['git_token'] == FRESH_TOKEN


def test_dataset_repository_push_sends_a_url_without_the_stale_token():
    wrapper = _wrapper()
    _push(_repository(DatasetRepository, wrapper))

    sent = wrapper.push.call_args.kwargs['url']
    assert 'STALE_TOKEN' not in sent
    assert sent == CLEAN_URL


def test_both_implementations_clean_the_url_the_same_way():
    urls = []
    for cls in (Repository, DatasetRepository):
        wrapper = _wrapper()
        _push(_repository(cls, wrapper))
        urls.append(wrapper.push.call_args.kwargs['url'])
    assert urls[0] == urls[1]


def test_repository_push_asks_the_wrapper_to_strip_the_token():
    wrapper = _wrapper()
    _push(_repository(Repository, wrapper))
    # The strip has to happen through the wrapper (which knows the token
    # formats), not by string surgery at the call site.
    wrapper.remove_token_from_url.assert_called_once_with(STALE_URL)
