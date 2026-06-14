# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression test for Issue #1668.

`TinynasClassificationPipeline.postprocess` previously called `eval()` on
the raw contents of `label_map.txt` shipped from the remote model repo,
giving attackers RCE the moment the pipeline produced a prediction. The fix
swaps `eval` for `ast.literal_eval`.
"""
import os
import shutil
import sys
import tempfile
import types
import unittest

import torch  # noqa: E402

from modelscope.pipelines.cv.tinynas_classification_pipeline import \
    TinynasClassificationPipeline  # noqa: E402

# `tinynas_classification_pipeline` imports torch/torchvision and our own
# `tinynas_classfication` module at top level. Stub the latter only when
# missing so we don't drag heavy custom CUDA ops into the test.
if 'modelscope.models.cv.tinynas_classfication' not in sys.modules:
    try:
        from modelscope.models.cv.tinynas_classfication import \
            get_zennet  # noqa: F401
    except Exception:
        stub = types.ModuleType('modelscope.models.cv.tinynas_classfication')
        stub.get_zennet = lambda *a, **kw: None
        sys.modules['modelscope.models.cv.tinynas_classfication'] = stub


class TinynasLabelMapSecurityTest(unittest.TestCase):
    """Pin the `ast.literal_eval` gate around remote `label_map.txt`."""

    def setUp(self):
        self.tmp_root = tempfile.mkdtemp(prefix='ms_tinynas_security_')
        self.label_path = os.path.join(self.tmp_root, 'label_map.txt')
        # Bypass __init__ — it would try to load a real checkpoint.
        self.pipe = TinynasClassificationPipeline.__new__(
            TinynasClassificationPipeline)
        self.pipe.path = self.tmp_root

    def tearDown(self):
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def _write(self, content):
        with open(self.label_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def _fake_inputs(self, argmax_idx):
        # `outputs` must support `.argmax().item()` and softmax.
        outputs = torch.zeros(1, 3)
        outputs[0, argmax_idx] = 10.0
        return {'outputs': outputs}

    def test_legitimate_label_map_parses(self):
        self._write("{0: 'cat', 1: 'dog', 2: 'wolf'}")
        out = self.pipe.postprocess(self._fake_inputs(2))
        self.assertEqual(out['labels'], ['wolf'])
        self.assertEqual(len(out['scores']), 1)

    def test_malicious_import_payload_blocked(self):
        canary = '/tmp/ms_tinynas_rce_canary'
        if os.path.exists(canary):
            os.remove(canary)
        self._write(
            "__import__('os').system('echo HACKED && touch {}')".format(
                canary))
        with self.assertRaises((ValueError, SyntaxError)):
            self.pipe.postprocess(self._fake_inputs(0))
        self.assertFalse(
            os.path.exists(canary),
            'os.system was reached -- eval gate is bypassed.')

    def test_malicious_function_call_blocked(self):
        self._write("print('pwned')")
        with self.assertRaises((ValueError, SyntaxError)):
            self.pipe.postprocess(self._fake_inputs(0))

    def test_malicious_subclass_escape_blocked(self):
        self._write('().__class__.__bases__[0].__subclasses__()')
        with self.assertRaises((ValueError, SyntaxError)):
            self.pipe.postprocess(self._fake_inputs(0))

    def test_malformed_label_map_raises_cleanly(self):
        # Empty file is not a literal container; must not silently no-op.
        self._write('')
        with self.assertRaises((ValueError, SyntaxError)):
            self.pipe.postprocess(self._fake_inputs(0))


if __name__ == '__main__':
    unittest.main()
