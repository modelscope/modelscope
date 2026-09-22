# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modelscope.hub.utils.utils import find_reusable_legacy_repo_dir


class LegacyCacheReuseTest(unittest.TestCase):
    """Old flat/hub cache layouts should be reusable without re-download."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmpdir.name)
        self.model_id = 'iic/nlp_xlmr_named-entity-recognition_eng-ecommerce-query'
        self.owner, self.name = self.model_id.split('/', 1)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _touch_model_dir(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        (path / 'configuration.json').write_text('{}', encoding='utf-8')

    def test_finds_flat_legacy_cache(self):
        legacy = self.cache / self.owner / self.name
        self._touch_model_dir(legacy)
        found = find_reusable_legacy_repo_dir(
            self.model_id, cache_dir=self.cache)
        self.assertEqual(found, str(legacy))

    def test_finds_hub_legacy_cache(self):
        legacy = self.cache / 'hub' / self.owner / self.name
        self._touch_model_dir(legacy)
        found = find_reusable_legacy_repo_dir(
            self.model_id, cache_dir=self.cache)
        self.assertEqual(found, str(legacy))

    def test_reuses_unsafed_models_slash_layout(self):
        # Hub only probes safe_name (dots -> ___); unsafed path is reusable.
        dotted_id = 'org/model.with.dots'
        owner, name = dotted_id.split('/', 1)
        slash = self.cache / 'models' / owner / name
        self._touch_model_dir(slash)
        found = find_reusable_legacy_repo_dir(dotted_id, cache_dir=self.cache)
        self.assertEqual(found, str(slash))

    def test_prefers_hub_known_safe_slash_layout(self):
        dotted_id = 'org/model.with.dots'
        owner, name = dotted_id.split('/', 1)
        safe = self.cache / 'models' / owner / name.replace('.', '___')
        flat = self.cache / owner / name
        self._touch_model_dir(safe)
        self._touch_model_dir(flat)
        found = find_reusable_legacy_repo_dir(dotted_id, cache_dir=self.cache)
        self.assertIsNone(found)

    def test_prefers_hub_known_owner_dash_layout(self):
        modern = self.cache / 'models' / self.model_id.replace('/', '--')
        flat = self.cache / self.owner / self.name
        self._touch_model_dir(modern)
        self._touch_model_dir(flat)
        found = find_reusable_legacy_repo_dir(
            self.model_id, cache_dir=self.cache)
        self.assertIsNone(found)

    def test_empty_legacy_dir_ignored(self):
        (self.cache / self.owner / self.name).mkdir(parents=True)
        found = find_reusable_legacy_repo_dir(
            self.model_id, cache_dir=self.cache)
        self.assertIsNone(found)

    def test_uses_modelscope_cache_env(self):
        legacy = self.cache / self.owner / self.name
        self._touch_model_dir(legacy)
        with mock.patch.dict(os.environ,
                             {'MODELSCOPE_CACHE': str(self.cache)}):
            found = find_reusable_legacy_repo_dir(self.model_id)
        self.assertEqual(found, str(legacy))

    def test_default_root_matches_hub_not_sdk_hub_suffix(self):
        # Without MODELSCOPE_CACHE, hub uses ~/.cache/modelscope (no /hub).
        modern = (
            Path.home() / '.cache' / 'modelscope' / 'models'
            / self.model_id.replace('/', '--'))
        # Do not create real home dirs; patch the hub root helper instead.
        with mock.patch(
                'modelscope.hub.utils.utils._modelscope_hub_cache_root',
                return_value=self.cache):
            modern_under_test = (
                self.cache / 'models' / self.model_id.replace('/', '--'))
            flat = self.cache / self.owner / self.name
            self._touch_model_dir(modern_under_test)
            self._touch_model_dir(flat)
            found = find_reusable_legacy_repo_dir(self.model_id)
        self.assertIsNone(found)
        self.assertFalse(modern.exists())  # we never touched real home cache


if __name__ == '__main__':
    unittest.main()
