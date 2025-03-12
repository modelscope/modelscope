# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

from requests import HTTPError

from modelscope import MsDataset, snapshot_download
from modelscope.hub.constants import MODELSCOPE_PREFER_AI_SITE


class HubAiSiteTest(unittest.TestCase):

    def setUp(self):
        ...

    # test download from an ai-site only model, it should
    # work as expected since we shall fall back to ai-site
    # when the model is not found on cn-site.
    def test_default_download_from_ai_site(self):
        model_id = 'ModelScope_Developer/ai_only'
        model_dir = snapshot_download(model_id)
        contents = os.listdir(model_dir)
        assert len(contents) > 0

    # test download from a cn-site only model, it should
    # work as expected as it is found directly on cn-site.
    def test_default_download_from_cn_site(self):
        model_id = 'ModelScope_Developer/cn_only'
        model_dir = snapshot_download(model_id)
        contents = os.listdir(model_dir)
        assert len(contents) > 0

    # test download a model that exists on both cn and ai site
    # when prefer-ai-site is set, we should found the version from
    # on ai-site, not cn-site
    def test_prefer_ai_site_and_download_from_ai_site(self):
        os.environ[MODELSCOPE_PREFER_AI_SITE] = 'True'
        model_id = 'ModelScope_Developer/same_name'
        model_dir = snapshot_download(model_id)
        cn_site_only_file = os.path.join(model_dir, 'on_ai_site')
        assert os.path.exists(cn_site_only_file)

    # test download a model that exists on both cn and ai site
    # when prefer-ai-site is NOT set, we should found the version from
    # on cn-site, not ai-site
    def test_prefer_cn_site_and_download_from_cn_site(self):
        os.environ[MODELSCOPE_PREFER_AI_SITE] = 'False'
        model_id = 'ModelScope_Developer/same_name'
        model_dir = snapshot_download(model_id)
        cn_site_only_file = os.path.join(model_dir, 'on_cn_site')
        assert os.path.exists(cn_site_only_file)

    def test_download_non_exist_model(self):
        with self.assertRaises(HTTPError):
            model_id = 'ModelScope_Developer/not_exist_model'
            snapshot_download(model_id)

    # test download dataset from ai site
    def test_download_dataset_from_ai_site(self):
        os.environ[MODELSCOPE_PREFER_AI_SITE] = 'True'
        dataset_id = 'ModelScope_Developer/ai_only_dataset'
        dataset = MsDataset.load(dataset_id)
        assert dataset


if __name__ == '__main__':
    unittest.main()
