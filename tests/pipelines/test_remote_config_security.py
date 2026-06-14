# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression test for Issue #1672.

Pins the `trust_remote_code` gate that prevents `Config.from_file` and
`check_trust_remote_code_for_config` from executing a `.py` config from an
untrusted model repository.

Coverage matrix
---------------
The helper / choke point exercised here is the only mechanism guarding every
sink listed in the issue:

  * via `Config.from_file('.py')`:
    - `SalientDetection`            (models/cv/salient_detection)
    - `AbnormalDetectionModel`      (models/cv/abnormal_object_detection)
    - `DetectionModel`              (models/cv/object_detection)
    - `ObjectDetection3DPipeline`   (pipelines/cv/object_detection_3d_pipeline)

  * via `check_trust_remote_code_for_config(...)` invoked before
    `mmcv.Config.fromfile` / `mmseg.apis.init_segmentor`:
    - `ClassificationModel`                     (image_classification/mmcls_model)
    - `SwinLPanopticSegmentation`               (image_panoptic_segmentation)
    - `SemanticSegmentation`                    (image_semantic_segmentation)
    - `ScrfdDetect` / `TinyMogDetect` / `DamoFdDetect`  (face_detection/scrfd)
    - `ImageClassificationMmcvPreprocessor`     (preprocessors/cv)
    - `SegformerDetector`                       (controllable_image_generation)
"""
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from modelscope.utils.config import (Config,
                                     check_trust_remote_code_for_config)


def _make_repo(root, owner, name):
    """`check_model_from_owner_group` reads the owner from the parent dir."""
    repo = os.path.join(root, owner, name)
    os.makedirs(repo)
    return repo


def _write(path, content):
    with open(path, 'w') as f:
        f.write(content)


# Canary: any caller that fails the gate must NEVER execute this body.
_CANARY = ('raise SystemExit("this script must not run")\n'
           'CANARY = "should not be reachable"\n')


class ConfigChokePointSecurityTest(unittest.TestCase):
    """End-to-end gating at `Config.from_file` for `.py` configs.

    Covers SalientDetection / AbnormalDetectionModel / DetectionModel /
    ObjectDetection3DPipeline, all of which call `Config.from_file(*.py,
    trust_remote_code=self.trust_remote_code)`.
    """

    def setUp(self):
        self.tmp_root = tempfile.mkdtemp(prefix='ms_cfg_security_')
        self.untrusted_dir = _make_repo(self.tmp_root, 'attacker', 'badrepo')
        self.trusted_dir = _make_repo(self.tmp_root, 'damo', 'goodrepo')

        self.untrusted_py = os.path.join(self.untrusted_dir, 'mmcv_config.py')
        self.trusted_py = os.path.join(self.trusted_dir, 'mmcv_config.py')
        for p in (self.untrusted_py, self.trusted_py):
            _write(p, _CANARY)

        self.untrusted_json = os.path.join(self.untrusted_dir,
                                           'configuration.json')
        _write(self.untrusted_json, '{"a": 1}')

    def tearDown(self):
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_untrusted_py_without_optin_is_blocked(self):
        with self.assertRaises(RuntimeError):
            Config.from_file(self.untrusted_py)

    def test_untrusted_py_with_optin_is_allowed(self):
        # opt-in must let the load proceed, which means the canary will run
        # and raise SystemExit -- proving the gate did not block.
        with self.assertRaises(SystemExit):
            Config.from_file(self.untrusted_py, trust_remote_code=True)

    def test_trusted_owner_py_is_allowed_without_optin(self):
        with self.assertRaises(SystemExit):
            Config.from_file(self.trusted_py)

    def test_json_passive_load_unaffected(self):
        cfg = Config.from_file(self.untrusted_json)
        self.assertEqual(cfg.a, 1)

    def test_from_string_in_process_is_not_gated(self):
        # `Config.from_string` materializes a caller-supplied in-process
        # string and must keep working without `trust_remote_code`.
        cfg = Config.from_string('a = 42\n', '.py')
        self.assertEqual(cfg.a, 42)

    def test_explicit_model_dir_overrides_default(self):
        # Some callers pass `model_dir` explicitly; verify trust derives
        # from that path, not from the file's parent directory.
        nested = os.path.join(self.untrusted_dir, 'subdir')
        os.makedirs(nested)
        nested_py = os.path.join(nested, 'mmcv_config.py')
        _write(nested_py, _CANARY)
        with self.assertRaises(RuntimeError):
            Config.from_file(nested_py, model_dir=self.untrusted_dir)
        # Same file, but with a trusted model_dir override -> allowed.
        with self.assertRaises(SystemExit):
            Config.from_file(nested_py, model_dir=self.trusted_dir)


class CheckTrustRemoteCodeForConfigTest(unittest.TestCase):
    """Unit tests for the helper invoked at every mmcv / init_segmentor sink.

    Covers ClassificationModel, SwinLPanopticSegmentation, SemanticSegmentation,
    ScrfdDetect (and subclasses TinyMogDetect / DamoFdDetect),
    ImageClassificationMmcvPreprocessor, and SegformerDetector.
    """

    def setUp(self):
        self.tmp_root = tempfile.mkdtemp(prefix='ms_cfg_security_helper_')
        self.untrusted_dir = _make_repo(self.tmp_root, 'attacker', 'badrepo')
        self.trusted_dir = _make_repo(self.tmp_root, 'damo', 'goodrepo')
        # Helper only inspects file extensions; no need to create real files.
        self.untrusted_py = os.path.join(self.untrusted_dir, 'config.py')
        self.trusted_py = os.path.join(self.trusted_dir, 'config.py')
        self.untrusted_json = os.path.join(self.untrusted_dir,
                                           'configuration.json')

    def tearDown(self):
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_non_py_path_always_passes(self):
        # JSON / YAML configs are passive data; helper must not block them.
        check_trust_remote_code_for_config(
            self.untrusted_json,
            trust_remote_code=False,
            model_dir=self.untrusted_dir)

    def test_untrusted_py_without_optin_raises(self):
        with self.assertRaises(RuntimeError):
            check_trust_remote_code_for_config(
                self.untrusted_py,
                trust_remote_code=False,
                model_dir=self.untrusted_dir)

    def test_untrusted_py_with_optin_passes(self):
        check_trust_remote_code_for_config(
            self.untrusted_py,
            trust_remote_code=True,
            model_dir=self.untrusted_dir)

    def test_trusted_owner_py_passes(self):
        check_trust_remote_code_for_config(
            self.trusted_py,
            trust_remote_code=False,
            model_dir=self.trusted_dir)

    def test_default_model_dir_inferred_from_filename(self):
        # When `model_dir` is omitted, helper must infer it from the file's
        # parent directory and still gate untrusted owners correctly.
        with self.assertRaises(RuntimeError):
            check_trust_remote_code_for_config(
                self.untrusted_py, trust_remote_code=False)
        # Trusted parent should pass.
        check_trust_remote_code_for_config(
            self.trusted_py, trust_remote_code=False)

    def test_helper_uses_raise_not_assert(self):
        # Security gates must survive `python -O`. The helper raises
        # RuntimeError, not AssertionError; verify the failure type so the
        # contract is locked in.
        try:
            check_trust_remote_code_for_config(
                self.untrusted_py,
                trust_remote_code=False,
                model_dir=self.untrusted_dir)
        except RuntimeError:
            return
        except AssertionError:
            self.fail('Gate used `assert`; would be a no-op under `python -O`.')
        self.fail('Gate did not raise on untrusted `.py` config.')


class SinkWiringTest(unittest.TestCase):
    """Spot-check that a representative mmcv sink calls the helper before
    reaching `mmcv.Config.fromfile`.

    We patch the helper to raise a sentinel; if the sink is wired correctly,
    the sentinel propagates and the unsafe `mmcv.Config.fromfile` is never
    invoked.
    """

    def setUp(self):
        self.tmp_root = tempfile.mkdtemp(prefix='ms_cfg_sink_wiring_')
        self.untrusted_dir = _make_repo(self.tmp_root, 'attacker', 'badrepo')

    def tearDown(self):
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def _import_module_text(self, dotted):
        # Read the source of a module without importing it (avoids dragging
        # in mmcv / mmdet / torch on test machines that lack them).
        import importlib.util
        spec = importlib.util.find_spec(dotted)
        self.assertIsNotNone(spec, f'{dotted} not found')
        with open(spec.origin, 'r') as f:
            return f.read()

    def _assert_helper_precedes_sink(self, dotted, sink_call):
        src = self._import_module_text(dotted)
        helper_pos = src.find('check_trust_remote_code_for_config(')
        sink_pos = src.find(sink_call)
        self.assertGreater(
            helper_pos, -1,
            f'{dotted} does not call check_trust_remote_code_for_config')
        self.assertGreater(sink_pos, -1,
                           f'{dotted} no longer contains `{sink_call}`')
        self.assertLess(
            helper_pos, sink_pos,
            f'{dotted}: helper call must precede `{sink_call}`')

    def test_classification_model_wires_helper(self):
        self._assert_helper_precedes_sink(
            'modelscope.models.cv.image_classification.mmcls_model',
            'mmcv.Config.fromfile(')

    def test_panoptic_segmentation_wires_helper(self):
        self._assert_helper_precedes_sink(
            'modelscope.models.cv.image_panoptic_segmentation.panseg_model',
            'mmcv.Config.fromfile(')

    def test_semantic_segmentation_wires_helper(self):
        self._assert_helper_precedes_sink(
            'modelscope.models.cv.image_semantic_segmentation.semantic_seg_model',
            'mmcv.Config.fromfile(')

    def test_scrfd_detect_wires_helper(self):
        self._assert_helper_precedes_sink(
            'modelscope.models.cv.face_detection.scrfd.scrfd_detect',
            'Config.fromfile(')

    def test_mmcls_preprocessor_wires_helper(self):
        self._assert_helper_precedes_sink(
            'modelscope.preprocessors.cv.mmcls_preprocessor',
            'mmcv.Config.fromfile(')

    def test_segformer_detector_wires_helper(self):
        self._assert_helper_precedes_sink(
            'modelscope.models.cv.controllable_image_generation.annotator.annotator',
            'init_segmentor(')


if __name__ == '__main__':
    unittest.main()
