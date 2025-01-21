import os
from typing import Dict

import torch

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.compatible_with_transformers import \
    compatible_position_ids
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from .backbone import Re2GModel


@MODELS.register_module(
    Tasks.document_grounded_dialog_generate, module_name=Models.doc2bot)
class DocumentGroundedDialogGenerateModel(TorchModel):
    _backbone_prefix = ''

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.config = Config.from_file(
            os.path.join(self.model_dir, ModelFile.CONFIGURATION))
        self.model = Re2GModel(model_dir, self.config)
        state_dict = torch.load(
            os.path.join(self.model_dir, ModelFile.TORCH_MODEL_BIN_FILE),
            map_location='cpu')
        compatible_position_ids(
            state_dict, 'rerank.encoder.roberta.embeddings.position_ids')
        self.model.load_state_dict(state_dict)

    def forward(self, input: Dict[str, Tensor]):
        rerank_input_ids = input['rerank_input_ids']
        input_ids = input['input_ids']
        attention_mask = input['attention_mask']
        label_ids = input['label_ids']

        outputs = self.model(rerank_input_ids, input_ids, attention_mask,
                             label_ids)
        return outputs

    def generate(self, input: Dict[str, Tensor]):
        rerank_input_ids = input['rerank_input_ids']
        input_ids = input['input_ids']
        attention_mask = input['attention_mask']
        outputs = self.model.generate(rerank_input_ids, input_ids,
                                      attention_mask)
        return outputs
