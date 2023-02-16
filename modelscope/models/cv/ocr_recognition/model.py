# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import torch
import torch.nn.functional as F

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .modules.convnextvit import ConvNextViT
from .modules.crnn import CRNN

LOGGER = get_logger()


@MODELS.register_module(
    Tasks.ocr_recognition, module_name=Models.ocr_recognition)
class OCRRecognition(TorchModel):

    def __init__(self, model_dir: str, **kwargs):
        """initialize the ocr recognition model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, **kwargs)

        model_path = os.path.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        cfgs = Config.from_file(
            os.path.join(model_dir, ModelFile.CONFIGURATION))
        self.do_chunking = cfgs.model.inference_kwargs.do_chunking
        self.recognizer = None
        if cfgs.model.recognizer == 'ConvNextViT':
            self.recognizer = ConvNextViT()
        elif cfgs.model.recognizer == 'CRNN':
            self.recognizer = CRNN()
        else:
            raise TypeError(
                f'recognizer should be either ConvNextViT, CRNN, but got {cfgs.model.recognizer}'
            )
        if model_path != '':
            self.recognizer.load_state_dict(
                torch.load(model_path, map_location='cpu'))

        dict_path = os.path.join(model_dir, ModelFile.VOCAB_FILE)
        self.labelMapping = dict()
        with open(dict_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            cnt = 1
            for line in lines:
                line = line.strip('\n')
                self.labelMapping[cnt] = line
                cnt += 1

    def forward(self, inputs):
        """
        Args:
            img (`torch.Tensor`): batched image tensor,
                shape of each tensor is [N, 1, H, W].

        Return:
            `probs [T, N, Classes] of the sequence feature`
        """
        return self.recognizer(inputs)

    def postprocess(self, inputs):
        # naive decoder
        if self.do_chunking:
            preds = inputs
            batchSize, length = preds.shape
            PRED_LENTH = 75
            PRED_PAD = 6
            pred_idx = []
            if batchSize == 1:
                pred_idx = preds[0].cpu().data.tolist()
            else:
                for idx in range(batchSize):
                    if idx == 0:
                        pred_idx.extend(
                            preds[idx].cpu().data[:PRED_LENTH
                                                  - PRED_PAD].tolist())
                    elif idx == batchSize - 1:
                        pred_idx.extend(
                            preds[idx].cpu().data[PRED_PAD:].tolist())
                    else:
                        pred_idx.extend(
                            preds[idx].cpu().data[PRED_PAD:PRED_LENTH
                                                  - PRED_PAD].tolist())
            pred_idx = [its - 1 for its in pred_idx if its > 0]
        else:
            outprobs = inputs
            outprobs = F.softmax(outprobs, dim=-1)
            preds = torch.argmax(outprobs, -1)
            length, batchSize = preds.shape
            assert batchSize == 1, 'only support onesample inference'
            pred_idx = preds[:, 0].cpu().data.tolist()

        pred_idx = pred_idx
        last_p = 0
        str_pred = []
        for p in pred_idx:
            if p != last_p and p != 0:
                str_pred.append(self.labelMapping[p])
            last_p = p
        final_str = ''.join(str_pred)
        return final_str
