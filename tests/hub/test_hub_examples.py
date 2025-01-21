# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.hub.api import HubApi
from modelscope.utils.hub import create_model_if_not_exist

# note this is temporary before official account management is ready
YOUR_ACCESS_TOKEN = 'Get SDK token from https://www.modelscope.cn/my/myaccesstoken'


class HubExampleTest(unittest.TestCase):

    def setUp(self):
        self.api = HubApi()
        self.api.login(YOUR_ACCESS_TOKEN)

    @unittest.skip('to be used for local test only')
    def test_example_model_creation(self):
        # ATTENTION:change to proper model names before use
        model_name = 'model-name'
        model_chinese_name = '我的测试模型'
        model_owner = 'iic'
        model_id = '%s/%s' % (model_owner, model_name)
        created = create_model_if_not_exist(self.api, model_id,
                                            model_chinese_name)
        if not created:
            print('!! NOT created since model already exists !!')


if __name__ == '__main__':
    unittest.main()
