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
from .modules.ConvNextViT.main_model import ConvNextViT
from .modules.CRNN.main_model import CRNN
from .modules.LightweightEdge.main_model import LightweightEdge

LOGGER = get_logger()


def flatten_label(target):
    label_flatten = []
    label_length = []
    label_dict = []
    for i in range(0, target.size()[0]):
        cur_label = target[i].tolist()
        temp_label = cur_label[:cur_label.index(0)]
        label_flatten += temp_label
        label_dict.append(temp_label)
        label_length.append(len(temp_label))
    label_flatten = torch.LongTensor(label_flatten)
    label_length = torch.IntTensor(label_length)
    return (label_dict, label_length, label_flatten)


class cha_encdec():

    def __init__(self, charMapping, case_sensitive=True):
        self.case_sensitive = case_sensitive
        self.text_seq_len = 160
        self.charMapping = charMapping

    def encode(self, label_batch):
        max_len = max([len(s) for s in label_batch])
        out = torch.zeros(len(label_batch), max_len + 1).long()
        for i in range(0, len(label_batch)):
            if not self.case_sensitive:
                cur_encoded = torch.tensor([
                    self.charMapping[char.lower()] - 1 if char.lower()
                    in self.charMapping else len(self.charMapping)
                    for char in label_batch[i]
                ]) + 1
            else:
                cur_encoded = torch.tensor([
                    self.charMapping[char]
                    - 1 if char in self.charMapping else len(self.charMapping)
                    for char in label_batch[i]
                ]) + 1
            out[i][0:len(cur_encoded)] = cur_encoded
        out = torch.cat(
            (out, torch.zeros(
                (out.size(0), self.text_seq_len - out.size(1))).type_as(out)),
            dim=1)
        label_dict, label_length, label_flatten = flatten_label(out)
        return label_dict, label_length, label_flatten


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
        self.target_height = cfgs.model.inference_kwargs.img_height
        self.target_width = cfgs.model.inference_kwargs.img_width
        self.recognizer = None
        if cfgs.model.recognizer == 'ConvNextViT':
            self.recognizer = ConvNextViT()
        elif cfgs.model.recognizer == 'CRNN':
            self.recognizer = CRNN()
        elif cfgs.model.recognizer == 'LightweightEdge':
            self.recognizer = LightweightEdge()
        else:
            raise TypeError(
                f'recognizer should be either ConvNextViT, CRNN, but got {cfgs.model.recognizer}'
            )
        if model_path != '':
            params_pretrained = torch.load(model_path, map_location='cpu')
            model_dict = self.recognizer.state_dict()
            # remove prefix for finetuned models
            check_point = {
                k.replace('recognizer.', '').replace('module.', ''): v
                for k, v in params_pretrained.items()
            }
            model_dict.update(check_point)
            self.recognizer.load_state_dict(model_dict)

        dict_path = os.path.join(model_dir, ModelFile.VOCAB_FILE)
        self.labelMapping = dict()
        self.charMapping = dict()
        with open(dict_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            cnt = 1
            # ConvNextViT and LightweightEdge model start from index=2
            if cfgs.model.recognizer == 'ConvNextViT' or cfgs.model.recognizer == 'LightweightEdge':
                cnt += 1
            for line in lines:
                line = line.strip('\n')
                self.labelMapping[cnt] = line
                self.charMapping[line] = cnt
                cnt += 1

        self.encdec = cha_encdec(self.charMapping)
        self.criterion_CTC = torch.nn.CTCLoss(zero_infinity=True)

    def forward(self, inputs):
        """
        Args:
            img (`torch.Tensor`): batched image tensor,
                shape of each tensor is [N, 1, H, W].

        Return:
            `probs [T, N, Classes] of the sequence feature`
        """
        return self.recognizer(inputs)

    def do_step(self, batch):
        inputs = batch['images']
        labels = batch['labels']
        bs = inputs.shape[0]
        if self.do_chunking:
            inputs = inputs.view(bs * 3, 3, self.target_height, 300)
        else:
            inputs = inputs.view(bs, 3, self.target_height, self.target_width)
        output = self(inputs)
        probs = output['probs'].permute(1, 0, 2)
        _, label_length, label_flatten = self.encdec.encode(labels)
        probs_sizes = torch.IntTensor([probs.size(0)] * probs.size(1))
        loss = self.criterion_CTC(
            probs.log_softmax(2), label_flatten, probs_sizes, label_length)
        output = dict(loss=loss, preds=output['preds'])
        return output

    def postprocess(self, inputs):
        outprobs = inputs
        outprobs = F.softmax(outprobs, dim=-1)
        preds = torch.argmax(outprobs, -1)
        batchSize, length = preds.shape
        final_str_list = []
        for i in range(batchSize):
            pred_idx = preds[i].cpu().data.tolist()
            last_p = 0
            str_pred = []
            for p in pred_idx:
                if p != last_p and p != 0:
                    str_pred.append(self.labelMapping[p])
                last_p = p
            final_str = ''.join(str_pred)
            final_str_list.append(final_str)
        return {'preds': final_str_list, 'probs': inputs}
