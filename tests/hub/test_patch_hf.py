import unittest

from modelscope.msdatasets import MsDataset
from modelscope.utils.test_utils import test_level


class DownloadDatasetTest(unittest.TestCase):

    def setUp(self):
        from modelscope.utils.hf_util import patch_hub
        patch_hub()

    def test_automodel_download(self):
        from transformers import AutoModel
        model = AutoModel.from_pretrained('AI-ModelScope/bert-base-uncased')
        self.assertTrue(model is not None)



