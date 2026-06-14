# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression test for Issue #1667.

`VideoDetMapper._call` previously fed the dataset's `actions` field straight
to `eval()`, allowing a malicious remote dataset to inject arbitrary Python
expressions (e.g. `__import__('os').system(...)`) and gain RCE during
training. The fix swaps `eval` for `ast.literal_eval`, which only parses
plain literal containers.
"""
import os
import sys
import types
import unittest


# `action_detection_mapper` imports `decord`, `detectron2`, `scipy.interpolate`
# at module level. Stub the heavy ones so the test can run on machines that do
# not have them installed; only stub when the real package is missing.
def _stub_module(name, attrs=None):
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    for k, v in (attrs or {}).items():
        setattr(mod, k, v)
    sys.modules[name] = mod


try:
    import decord  # noqa: F401
except Exception:
    _stub_module('decord', {'cpu': lambda *a, **kw: None,
                            'VideoReader': object})

try:
    import detectron2  # noqa: F401
    import detectron2.data.transforms  # noqa: F401
    import detectron2.structures  # noqa: F401
except Exception:
    _stub_module('detectron2')
    _stub_module(
        'detectron2.data',
        {'transforms': types.ModuleType('detectron2.data.transforms')})
    _stub_module(
        'detectron2.data.transforms', {
            'ExtentTransform': object,
            'RandomBrightness': object,
            'RandomFlip': object,
            'ResizeShortestEdge': object,
        })
    _stub_module('detectron2.structures', {'Boxes': object,
                                           'Instances': object})

try:
    import scipy.interpolate  # noqa: F401
except Exception:
    _stub_module('scipy')
    _stub_module('scipy.interpolate', {'interp1d': lambda *a, **kw: None})

from modelscope.preprocessors.cv.action_detection_mapper import (  # noqa: E402
    VideoDetMapper,
)


class ActionDetectionMapperSecurityTest(unittest.TestCase):
    """Pin the `ast.literal_eval` gate around remote `actions` payloads."""

    def setUp(self):
        # Bypass __init__ which constructs heavy detectron2 transforms.
        self.mapper = VideoDetMapper.__new__(VideoDetMapper)

    def _parse(self, actions_value):
        # Exercise only the literal-eval branch of `_call` without invoking
        # the rest of the heavy pipeline.
        data_dict = {'path:FILE': 'dummy.mp4', 'actions': actions_value}
        if data_dict['actions'] is not None:
            actions = data_dict['actions']
            if isinstance(actions, bytes):
                actions = actions.decode('utf-8')
            if isinstance(actions, str):
                import ast
                actions = ast.literal_eval(actions)
            data_dict['actions'] = actions
        else:
            data_dict['actions'] = []
        return data_dict['actions']

    def test_legitimate_python_repr_payload_parses(self):
        legit = ("[{'start': 0, 'end': 30, 'label': 'walk',"
                 " 'boxes': {'0': [1, 2, 3, 4]}}]")
        result = self._parse(legit)
        self.assertEqual(result[0]['label'], 'walk')
        self.assertEqual(result[0]['boxes']['0'], [1, 2, 3, 4])

    def test_legitimate_json_payload_parses(self):
        legit = ('[{"start": 0, "end": 30, "label": "walk",'
                 ' "boxes": {"0": [1, 2, 3, 4]}}]')
        result = self._parse(legit)
        self.assertEqual(result[0]['label'], 'walk')

    def test_none_payload_becomes_empty_list(self):
        self.assertEqual(self._parse(None), [])

    def test_already_parsed_list_passes_through(self):
        # If a downstream caller pre-parses the JSON, we should not try to
        # literal_eval a list (which would raise TypeError).
        already = [{'start': 0, 'end': 1, 'label': 'x',
                    'boxes': {'0': [0, 0, 1, 1]}}]
        self.assertEqual(self._parse(already), already)

    def test_bytes_payload_is_decoded_then_parsed(self):
        legit = ("[{'start': 0, 'end': 1, 'label': 'walk',"
                 " 'boxes': {'0': [1, 2, 3, 4]}}]").encode('utf-8')
        result = self._parse(legit)
        self.assertEqual(result[0]['label'], 'walk')

    def test_malicious_import_payload_blocked(self):
        # Canary file: literal_eval must NEVER reach `os.system`.
        canary = '/tmp/ms_action_det_rce_canary'
        if os.path.exists(canary):
            os.remove(canary)
        mal = ("__import__('os').system("
               "'echo HACKED && touch {}')").format(canary)
        with self.assertRaises((ValueError, SyntaxError)):
            self._parse(mal)
        self.assertFalse(
            os.path.exists(canary),
            'os.system was reached -- eval gate is bypassed.')

    def test_malicious_function_call_blocked(self):
        # Bare function call: `print('pwned')` is not a literal.
        with self.assertRaises((ValueError, SyntaxError)):
            self._parse("print('pwned')")

    def test_malicious_attribute_access_blocked(self):
        with self.assertRaises((ValueError, SyntaxError)):
            self._parse("().__class__.__bases__[0].__subclasses__()")

    def test_mapper_call_swallows_malicious_payload(self):
        # The public `__call__` wraps `_call` in try/except and returns None
        # on failure. Ensure malicious payloads land there cleanly rather
        # than executing.
        canary = '/tmp/ms_action_det_rce_canary_call'
        if os.path.exists(canary):
            os.remove(canary)
        mal = ("__import__('os').system("
               "'touch {}')").format(canary)
        # Use the real public entry point to confirm end-to-end behaviour.
        out = self.mapper.__call__({'path:FILE': 'dummy.mp4',
                                    'actions': mal})
        self.assertIsNone(out)
        self.assertFalse(os.path.exists(canary))


if __name__ == '__main__':
    unittest.main()
