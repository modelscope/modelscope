# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.test_utils import test_level


class BackboneTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.backbone
        self.model_id = 'damo/nlp_structbert_backbone_tiny_std'
        self.transformer_model = 'bert'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_load_backbone_model_with_ms_backbone(self):
        model = Model.from_pretrained(
            task=self.task, model_name_or_path=self.model_id)
        self.assertEqual(model.__class__.__name__, 'SbertModel')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_load_backbone_model_with_hf_automodel(self):
        local_model_dir = snapshot_download(self.model_id)
        cfg = Config.from_file(
            osp.join(local_model_dir, ModelFile.CONFIGURATION))
        cfg.model = {'type': 'transformers'}

        import json
        with open(osp.join(local_model_dir, ModelFile.CONFIG), 'r') as f:
            hf_config = json.load(f)

        hf_config['model_type'] = self.transformer_model

        with open(osp.join(local_model_dir, ModelFile.CONFIG), 'w') as f:
            json.dump(hf_config, f)

        model = Model.from_pretrained(
            task=self.task, model_name_or_path=self.model_id, cfg_dict=cfg)
        self.assertEqual(model.__class__.__name__, 'BertModel')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_load_backbone_model_with_hf_automodel_specific_model(self):
        self.transformer_model = 'roberta'

        local_model_dir = snapshot_download(self.model_id)
        cfg = Config.from_file(
            osp.join(local_model_dir, ModelFile.CONFIGURATION))
        cfg.model = {'type': self.transformer_model}
        import json
        with open(osp.join(local_model_dir, ModelFile.CONFIG), 'r') as f:
            hf_config = json.load(f)

        hf_config['model_type'] = self.transformer_model

        with open(osp.join(local_model_dir, ModelFile.CONFIG), 'w') as f:
            json.dump(hf_config, f)

        model = Model.from_pretrained(
            task=self.task, model_name_or_path=self.model_id, cfg_dict=cfg)
        self.assertEqual(model.__class__.__name__, 'RobertaModel')


if __name__ == '__main__':
    unittest.main()
