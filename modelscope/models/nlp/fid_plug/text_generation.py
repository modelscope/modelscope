# Copyright (c) Alibaba, Inc. and its affiliates.

import io
import os

import torch
from transformers.modeling_outputs import Seq2SeqLMOutput

from modelscope.metainfo import Models
from modelscope.models import Model
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import TextGenerationModelOutput, TokenGeneratorOutput
from modelscope.utils import logger as logging
from modelscope.utils.constant import Tasks
from .backbone import PlugForConditionalGeneration
from .configuration import PlugConfig

CONFIG_NAME = 'config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


class PlugV2Chat(TorchModel):

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        # init model
        plug_config_file = os.path.join(model_dir, CONFIG_NAME)
        plug_config = PlugConfig.from_json_file(plug_config_file)
        self.backbone = PlugForConditionalGeneration(plug_config)
        # load weights
        pretrained_model_path = os.path.join(model_dir, WEIGHTS_NAME)
        with io.open(pretrained_model_path, 'rb') as f:
            checkpoint = torch.load(f, map_location='cpu')
            if 'model' in checkpoint:
                checkpoint = checkpoint['model']
            for key in list(checkpoint.keys()):
                # for old plugv2 version
                if key.startswith('translator'):
                    checkpoint.pop(key)
                    continue
                if key.startswith('module.'):
                    checkpoint[key.replace('module.', '')] = checkpoint[key]
                    checkpoint.pop(key)
                if key.startswith('backbone.plug.bert.bert.'):
                    checkpoint[key.replace('backbone.plug.bert.bert.',
                                           'bert.')] = checkpoint[key]
                    checkpoint.pop(key)
                elif key.startswith('backbone.plug.'):
                    checkpoint[key.replace('backbone.plug.',
                                           '')] = checkpoint[key]
                    checkpoint.pop(key)
            msg = self.backbone.plug.load_state_dict(checkpoint, strict=False)
            print(f'| {msg}')

    def generate(self, input_ids, token_type_ids=None, *args, **kwargs):
        pred_result = self.backbone.translate(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            *args,
            **kwargs)['predictions']
        response = [x[0].tolist() for x in pred_result]
        response = torch.tensor(response)
        return response

    def forward(self,
                input_ids,
                decoder_input_ids,
                token_type_ids=None,
                *args,
                **kwargs):
        loss = self.backbone.forward(
            src=input_ids,
            tgt=decoder_input_ids,
            token_type_ids=token_type_ids,
            **kwargs)
        return Seq2SeqLMOutput(loss=loss[0], logits=loss[1])


class PlugV2EncoderWrapper(torch.nn.Module):

    def __init__(self, bert):
        super().__init__()

        self.bert = bert
        self.n_passages = None

    def set_n_passages(self, n_passages):
        self.n_passages = n_passages

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                *args,
                **kwargs):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz * self.n_passages, passage_length)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(bsz * self.n_passages,
                                                 passage_length)
        if attention_mask is not None:
            attention_mask = attention_mask.view(bsz * self.n_passages,
                                                 passage_length)
        outputs = self.bert(
            input_ids,
            attention_mask,
            token_type_ids=token_type_ids,
            *args,
            **kwargs)
        if isinstance(outputs, tuple):
            outputs = (outputs[0].view(bsz, self.n_passages * passage_length,
                                       -1), ) + outputs[1:]
        else:
            outputs.last_hidden_state = outputs.last_hidden_state.view(
                bsz, self.n_passages * passage_length, -1)
        return outputs


@MODELS.register_module(Tasks.fid_dialogue, module_name=Models.fid_plug)
class PlugV2FidChat(PlugV2Chat):

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.wrap_encoder()

    def wrap_encoder(self):
        self.backbone.plug.bert = PlugV2EncoderWrapper(self.backbone.plug.bert)

    def unwrap_encoder(self):
        self.backbone.plug.bert = self.backbone.plug.bert.bert

    def load(self,
             pretrained_model_path,
             from_tf=False):  # only invoked when model is not onnx format
        self.unwrap_encoder()
        super().load(pretrained_model_path)
        self.wrap_encoder()

    def generate(self, inputs, *args, **kwargs):
        input_ids = inputs.get('input_ids')
        token_type_ids = inputs.get('token_type_ids', None)
        n_passages = input_ids.size(1)
        self.backbone.plug.bert.set_n_passages(n_passages)
        input_ids = input_ids.view(input_ids.size(0), -1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(token_type_ids.size(0), -1)
        response = super().generate(
            input_ids, token_type_ids=token_type_ids, *args, **kwargs)
        return TokenGeneratorOutput(sequences=response)

    def forward(self,
                input_ids,
                decoder_input_ids,
                token_type_ids=None,
                *args,
                **kwargs):
        if input_ids is not None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                n_passages = input_ids.size(1)
                self.backbone.plug.bert.set_n_passages(n_passages)
            input_ids = input_ids.view(input_ids.size(0), -1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.view(input_ids.size(0), -1)
        seq2seq_lm_output = super().forward(
            input_ids,
            decoder_input_ids=decoder_input_ids,
            token_type_ids=token_type_ids,
            *args,
            **kwargs)
        return TextGenerationModelOutput(
            loss=seq2seq_lm_output.loss, logits=seq2seq_lm_output.logits)
