# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest
from typing import List, Union

from modelscope.utils.check_requirements import NLPModuleNotFoundError, get_msg
from modelscope.utils.constant import Fields


class ImportUtilsTest(unittest.TestCase):

    def test_type_module_not_found(self):
        with self.assertRaises(NLPModuleNotFoundError) as ctx:
            try:
                import not_found
            except ModuleNotFoundError as e:
                raise NLPModuleNotFoundError(e)
        self.assertTrue(get_msg(Fields.nlp) in ctx.exception.msg.msg)


if __name__ == '__main__':
    unittest.main()
