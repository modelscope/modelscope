# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression test for Issue #1673.

Verifies that Human3DAnimationPipeline.gen_weights() refuses to execute the
remote `skinning.py` script unless either (a) the model is owned by a trusted
group (`iic` / `damo`), or (b) the user explicitly opted in with
`trust_remote_code=True`.
"""
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

# `modelscope.models.cv.human3d_animation` lazily resolves to symbols that
# pull in heavy optional deps (e.g. `nvdiffrast`). We never invoke those
# symbols inside `gen_weights`, so substitute a lightweight stub *before* the
# pipeline module is imported. Only stub if the real package cannot be
# resolved, so we don't shadow it for environments that do have the deps.
_PKG = 'modelscope.models.cv.human3d_animation'
if _PKG not in sys.modules or not hasattr(sys.modules[_PKG], 'gen_skeleton_bvh'):
    try:
        from modelscope.models.cv.human3d_animation import (  # noqa: F401
            gen_skeleton_bvh, read_obj, write_obj,
        )
    except Exception:
        _stub = types.ModuleType(_PKG)
        _stub.gen_skeleton_bvh = lambda *a, **kw: None
        _stub.read_obj = lambda *a, **kw: None
        _stub.write_obj = lambda *a, **kw: None
        sys.modules[_PKG] = _stub

from modelscope.pipelines.cv.human3d_animation_pipeline import (  # noqa: E402
    Human3DAnimationPipeline,
)


class Human3DAnimationSecurityTest(unittest.TestCase):
    """Pin the trust_remote_code gate around remote `skinning.py` execution."""

    def setUp(self):
        self.tmp_root = tempfile.mkdtemp(prefix='ms_human3d_security_')
        # `check_model_from_owner_group` extracts the owner from the parent
        # directory name, so model dirs must live two levels deep.
        self.untrusted_dir = os.path.join(
            self.tmp_root, 'attacker', 'badrepo')
        self.trusted_dir = os.path.join(self.tmp_root, 'damo', 'goodrepo')
        for d in (self.untrusted_dir, self.trusted_dir):
            os.makedirs(d)
            with open(os.path.join(d, 'skinning.py'), 'w') as f:
                f.write("raise SystemExit('this script must not run')\n")

        self.case_dir = os.path.join(self.tmp_root, 'case')
        os.makedirs(self.case_dir)
        self.save_dir = os.path.join(self.tmp_root, 'out')

    def tearDown(self):
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def _make_pipeline(self, model_dir, trust_remote_code):
        # Bypass Pipeline.__init__ which would try to download a real model.
        p = Human3DAnimationPipeline.__new__(Human3DAnimationPipeline)
        p.model_dir = model_dir
        p.trust_remote_code = trust_remote_code
        p.case_dir = self.case_dir
        p.action = 'SwingDancing'
        p.blender = 'blender'
        return p

    def test_untrusted_repo_without_optin_is_blocked(self):
        p = self._make_pipeline(self.untrusted_dir, trust_remote_code=False)
        with mock.patch(
            'modelscope.pipelines.cv.human3d_animation_pipeline.os.system'
        ) as m_system:
            with self.assertRaises(RuntimeError):
                p.gen_weights(save_dir=self.save_dir)
            m_system.assert_not_called()

    def test_untrusted_repo_with_optin_is_allowed(self):
        p = self._make_pipeline(self.untrusted_dir, trust_remote_code=True)
        with mock.patch(
            'modelscope.pipelines.cv.human3d_animation_pipeline.os.system',
            return_value=0,
        ) as m_system:
            p.gen_weights(save_dir=self.save_dir)
            m_system.assert_called_once()

    def test_trusted_owner_is_allowed_without_optin(self):
        p = self._make_pipeline(self.trusted_dir, trust_remote_code=False)
        with mock.patch(
            'modelscope.pipelines.cv.human3d_animation_pipeline.os.system',
            return_value=0,
        ) as m_system:
            p.gen_weights(save_dir=self.save_dir)
            m_system.assert_called_once()


if __name__ == '__main__':
    unittest.main()
