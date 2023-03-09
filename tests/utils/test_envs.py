# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.utils.plugins import EnvsManager


class PluginTest(unittest.TestCase):

    def setUp(self):
        self.model_id = 'damo/nlp_nested-ner_named-entity-recognition_chinese-base-med'
        self.env_manager = EnvsManager(self.model_id)

    def tearDown(self):
        self.env_manager.clean_env()
        super().tearDown()

    def test_create_env(self):
        need_env = self.env_manager.check_if_need_env()
        self.assertEqual(need_env, True)
        activate_dir = self.env_manager.create_env()
        remote = 'source {}'.format(activate_dir)
        cmd = f'{remote};'
        print(cmd)
        # EnvsManager.run_process(cmd)  no sh in ci env, so skip
