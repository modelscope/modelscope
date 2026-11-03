# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for the `kernels` library monkey-patch in modelscope.

Requirements verified:

1. `from modelscope import get_kernel` works without `patch_hub()` first and
   routes downloads through ModelScope, without leaking the
   `_get_hf_api` patch to anyone else.
2. `patch_hub()` / `patch_context()` makes `from kernels import get_kernel`
   also route downloads through ModelScope; `unpatch_hub()` restores it.

The whole module requires `modelscope_hub.kernels` (`modelscope-hub>=0.7.0`);
it is skipped on older hubs where the kernels shims degrade to no-ops.
"""
import contextlib
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import pytest

pytest.importorskip("modelscope_hub.kernels")

from modelscope.utils.hf_util.patcher import (_patch_kernels, _unpatch_kernels,
                                              patch_context, patch_hub,
                                              unpatch_hub)


def _import_kernels_api_module():
    """Return the module holding `_get_hf_api` in the installed `kernels`.

    kernels >= 0.17 renamed `kernels.utils` to `kernels.hf_hub`; support both
    layouts for the original-factory assertions below.
    """
    try:
        import kernels.hf_hub as kernels_api
    except ImportError:
        import kernels.utils as kernels_api
    return kernels_api


def _ensure_kernels_installed():
    try:
        from kernels import get_kernel  # noqa: F401
        _import_kernels_api_module()
    except ImportError:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '-q', 'kernels'])
        for mod in list(sys.modules):
            if mod == 'kernels' or mod.startswith('kernels.'):
                sys.modules.pop(mod, None)


@contextlib.contextmanager
def _isolate_hub_patches():
    """Neutralize the non-kernels parts of `patch_hub()` so tests focus on
    the kernels monkey-patch behaviour only.
    """
    targets = [
        'modelscope.utils.hf_util.patcher._patch_hub',
        'modelscope.utils.hf_util.patcher._unpatch_hub',
        'modelscope.utils.hf_util.patcher._patch_pretrained_class',
        'modelscope.utils.hf_util.patcher._unpatch_pretrained_class',
    ]
    with contextlib.ExitStack() as stack:
        for t in targets:
            stack.enter_context(patch(t))
        stack.enter_context(
            patch(
                'modelscope.utils.hf_util.patcher.get_all_imported_modules',
                return_value=[]))
        yield


class _KernelsTestBase(unittest.TestCase):
    """Installs `kernels`, captures the original `_get_hf_api`, and keeps the
    state clean between tests.
    """

    @classmethod
    def setUpClass(cls):
        _ensure_kernels_installed()

    def setUp(self):
        _unpatch_kernels()
        # Instance attribute, so the descriptor protocol is not triggered.
        # kernels >= 0.17 renamed `kernels.utils` to `kernels.hf_hub`.
        kernels_api = _import_kernels_api_module()
        self.original_get_hf_api = kernels_api._get_hf_api
        self.kernels_utils = kernels_api

    def tearDown(self):
        _unpatch_kernels()
        import modelscope
        for name in ('get_kernel', 'has_kernel', 'install_kernel',
                     'load_kernel', 'get_locked_kernel', 'snapshot_download'):
            try:
                delattr(modelscope, name)
            except AttributeError:
                pass


class KernelsProxyApiTest(_KernelsTestBase):
    """Low-level proxy API behaviour exercised against the real
    `kernels.utils` / `kernels.hf_hub` module.
    """

    def _patched_api(self):
        _patch_kernels()
        return self.kernels_utils._get_hf_api()

    def test_patch_replaces_get_hf_api(self):
        api = self._patched_api()
        self.assertTrue(hasattr(self.kernels_utils, '_get_hf_api_origin'))
        # The patched API is the `modelscope_hub.kernels` shim, shared with
        # `from modelscope_hub.kernels import get_kernel`.
        from modelscope_hub.kernels._hub import MsKernelApi
        self.assertIsInstance(api, MsKernelApi)
        for name in ('snapshot_download', 'list_repo_tree', 'file_exists',
                     'list_repo_refs', 'hf_hub_download'):
            self.assertTrue(callable(getattr(api, name, None)), name)

    def test_patch_is_idempotent(self):
        _patch_kernels()
        first = self.kernels_utils._get_hf_api
        _patch_kernels()
        self.assertIs(self.kernels_utils._get_hf_api, first)

    def test_unpatch_restores_original(self):
        _patch_kernels()
        _unpatch_kernels()
        self.assertFalse(hasattr(self.kernels_utils, '_get_hf_api_origin'))
        self.assertIs(self.kernels_utils._get_hf_api, self.original_get_hf_api)

    def test_snapshot_download_routes_to_modelscope_hub(self):
        api = self._patched_api()
        with patch('modelscope_hub.HubApi') as hub_api_cls:
            api.snapshot_download(
                'foo/bar',
                allow_patterns=['build/*'],
                ignore_patterns=['*.md'],
                revision='main',
                max_workers=4)
        kwargs = hub_api_cls.return_value.download_repo.call_args.kwargs
        # `main` is normalized to `master`; kernel repos map onto model repos.
        self.assertEqual(hub_api_cls.return_value.download_repo.call_args.args,
                         ('foo/bar', 'model'))
        self.assertEqual(kwargs['revision'], 'master')
        self.assertEqual(kwargs['allow_patterns'], ['build/*'])
        self.assertEqual(kwargs['ignore_patterns'], ['*.md'])
        # Forwarding details (error mapping, ref synthesis, tree derivation)
        # are covered by the `modelscope_hub` test suite.


class PatchHubFlowTest(_KernelsTestBase):
    """Requirement 2: `patch_hub` / `patch_context` toggle the kernels patch."""

    def test_patch_hub_then_unpatch_hub_round_trip(self):
        with _isolate_hub_patches():
            self.assertIs(self.kernels_utils._get_hf_api,
                          self.original_get_hf_api)

            patch_hub()
            self.assertTrue(hasattr(self.kernels_utils, '_get_hf_api_origin'))
            self.assertTrue(
                hasattr(self.kernels_utils._get_hf_api(), 'snapshot_download'))

            unpatch_hub()
            self.assertFalse(hasattr(self.kernels_utils, '_get_hf_api_origin'))
            self.assertIs(self.kernels_utils._get_hf_api,
                          self.original_get_hf_api)

    def test_patch_context_round_trip(self):
        with _isolate_hub_patches():
            with patch_context():
                self.assertTrue(
                    hasattr(self.kernels_utils, '_get_hf_api_origin'))
            self.assertFalse(hasattr(self.kernels_utils, '_get_hf_api_origin'))


class ModelscopeImportTest(_KernelsTestBase):
    """Requirement 1: `from modelscope import get_kernel` delegates to the
    real `kernels.get_kernel`, scoping the patch to the wrapped call only.
    """

    def _fake_get_kernel_capturing_api(self, sink):
        """Build a fake `kernels.get_kernel` that records `_get_hf_api()` at
        call time, so the test can observe the patch state mid-call.
        """

        def _fake(*args, **kwargs):
            sink['api'] = self.kernels_utils._get_hf_api()
            sink['args'] = args
            sink['kwargs'] = kwargs
            return 'kernel-module'

        return _fake

    def test_from_modelscope_wraps_kernels_get_kernel(self):
        import kernels
        import modelscope

        ms_get_kernel = modelscope.get_kernel
        self.assertIsNot(ms_get_kernel, kernels.get_kernel)
        # Before any call, `_get_hf_api` stays the original.
        self.assertIs(self.kernels_utils._get_hf_api, self.original_get_hf_api)

        captured = {}
        with patch.object(kernels, 'get_kernel',
                          self._fake_get_kernel_capturing_api(captured)):
            result = ms_get_kernel('foo/bar', revision='v1')

        self.assertEqual(result, 'kernel-module')
        self.assertEqual(captured['args'], ('foo/bar', ))
        self.assertEqual(captured['kwargs'], {'revision': 'v1'})
        # Mid-call: ModelScope proxy was active.
        self.assertTrue(hasattr(captured['api'], 'snapshot_download'))
        # After the call: patch is rolled back.
        self.assertIs(self.kernels_utils._get_hf_api, self.original_get_hf_api)

    def test_wrapped_call_nests_inside_patch_hub(self):
        import kernels
        import modelscope
        captured = {}
        with _isolate_hub_patches():
            patch_hub()
            with patch.object(kernels, 'get_kernel',
                              self._fake_get_kernel_capturing_api(captured)):
                modelscope.get_kernel('foo/bar')

            # The outer `patch_hub` patch survives the wrapped call.
            self.assertTrue(hasattr(self.kernels_utils, '_get_hf_api_origin'))
            self.assertTrue(hasattr(captured['api'], 'snapshot_download'))

            unpatch_hub()
            self.assertIs(self.kernels_utils._get_hf_api,
                          self.original_get_hf_api)


class TinyGradRMSIntegrationTest(_KernelsTestBase):
    """Real-world check using `kernels-community/tinygrad-rms`, which is
    published on both HuggingFace and ModelScope. Verifies that the
    ModelScope download path actually works end to end.
    """

    REPO = 'kernels-community/tinygrad-rms'

    def test_from_modelscope_get_kernel(self):
        import modelscope
        # Routes through `try_import_from_hf` and scopes the ModelScope
        # monkey-patch to this single call.
        module = modelscope.get_kernel(
            self.REPO, trust_remote_code=True, revision='master')
        self.assertIsNotNone(module)
        # Wrapper must leave `_get_hf_api` restored afterwards.
        self.assertIs(self.kernels_utils._get_hf_api, self.original_get_hf_api)

    def test_patch_hub_then_kernels_get_kernel(self):
        with _isolate_hub_patches(), patch_context():
            from kernels import get_kernel
            module = get_kernel(
                self.REPO, trust_remote_code=True, revision='master')
            self.assertIsNotNone(module)
        # `patch_context` rolled the kernels patch back on exit.
        self.assertIs(self.kernels_utils._get_hf_api, self.original_get_hf_api)


if __name__ == '__main__':
    unittest.main()
