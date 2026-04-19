# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.utils.import_utils import exists


class CompatibilityTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def tearDown(self):
        super().tearDown()

    @unittest.skipUnless(
        exists('xtcocotools'), 'Skip because xtcocotools is not installed.')
    def test_xtcocotools(self):
        from xtcocotools.coco import COCO


if __name__ == '__main__':
    unittest.main()
