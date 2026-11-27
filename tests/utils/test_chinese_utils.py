# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.utils.chinese_utils import (normalize_chinese_number,
                                            pre_chinese)
from modelscope.utils.import_utils import is_package_available
from modelscope.utils.test_utils import test_level


class ChineseUtilsTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    @unittest.skipUnless(
        is_package_available('zhconv'), 'zhconv is not installed')
    def test_normalize_chinese_number_maps_each_digit(self):
        self.assertEqual(normalize_chinese_number('0123456789'), '零一二三四五六七八九')
        self.assertEqual(normalize_chinese_number('我有2024个苹果'), '我有二零二四个苹果')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_pre_chinese_removes_punctuation(self):
        self.assertEqual(
            pre_chinese('你好，世界！Hello, World.', 100), '你好 世界 hello world')
        self.assertEqual(pre_chinese('“今天”天气（很）好。', 100), '今天 天气 很 好')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_pre_chinese_truncates_to_max_words(self):
        self.assertEqual(pre_chinese('你好，世界！', 4), '你好 世')


if __name__ == '__main__':
    unittest.main()
