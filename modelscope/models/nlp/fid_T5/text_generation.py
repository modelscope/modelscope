# Copyright (c) Alibaba, Inc. and its affiliates.

import io
import os

import torch
from transformers import AutoConfig

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.nlp.T5 import T5ForConditionalGeneration
from modelscope.outputs import TextGenerationModelOutput, TokenGeneratorOutput
from modelscope.utils.constant import Tasks

WEIGHTS_NAME = 'pytorch_model.bin'


class T5Chat(TorchModel):

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        # init model
        config = AutoConfig.from_pretrained(model_dir)
        self.backbone = T5ForConditionalGeneration(config)
        # load weights
        pretrained_model_path = os.path.join(model_dir, WEIGHTS_NAME)
        with io.open(pretrained_model_path, 'rb') as f:
            print('before torch.load..')
            state_dict = torch.load(f, map_location='cpu')
            state_dict = {
                k.replace('module.', ''): v
                for k, v in state_dict.items()
            }
            print('after torch.load..')
            model_state_dict = self.backbone.state_dict()
            model_keys = model_state_dict.keys()

            for old_name in list(state_dict.keys()):
                if old_name.startswith('module.'):
                    new_name = old_name.replace('module.', '')
                    state_dict[new_name] = state_dict.pop(old_name)
                    old_name = new_name
                if old_name.startswith('backbone.encoder.encoder'):
                    # wrap encoder name map
                    new_name = old_name.replace('backbone.encoder.encoder',
                                                'encoder').replace(
                                                    'module.layer', 'layer')
                    state_dict[new_name] = state_dict.pop(old_name)
                elif old_name.startswith('backbone'):
                    new_name = old_name[old_name.index('.')
                                        + 1:]  # remove backbone. prefix
                    state_dict[new_name] = state_dict.pop(old_name)

            missing_keys = [key for key in model_keys if key not in state_dict]
            unexpected_keys = [
                key for key in state_dict if key not in model_keys
            ]
            mismatched_keys = [
                key for key in model_keys if key in state_dict
                and state_dict[key].shape != model_state_dict[key].shape
            ]
            for key in mismatched_keys:
                del state_dict[key]

            self.backbone.load_state_dict(state_dict, strict=False)
            print(
                f'| Weights loaded for {self.backbone.__class__.__name__} from {pretrained_model_path}.'
            )
            if missing_keys:
                print('Missing keys:\n|\t' + '\n|\t'.join(missing_keys))
            if unexpected_keys:
                print('Unexpected keys:\n|\t' + '\n|\t'.join(unexpected_keys))
            if mismatched_keys:
                print('Mismatched keys:\n|\t' + '\n|\t'.join(mismatched_keys))
        # self.backbone.half()

    def generate(self, input_ids, *args, **kwargs) -> TokenGeneratorOutput:
        self.backbone.eval()
        with torch.no_grad():
            response = self.backbone.generate(input_ids, *args, **kwargs)
        return response

    def forward(self, input_ids, decoder_input_ids, *args, **kwargs):
        attention_mask = input_ids.ne(0).long()
        return self.backbone.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=decoder_input_ids,
            *args,
            **kwargs)


class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """

    def __init__(self, encoder):
        super().__init__()
        self.main_input_name = 'input_ids'
        self.encoder = encoder
        self.n_passages = None

    def set_n_passages(self, n_passages):
        self.n_passages = n_passages

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        *args,
        **kwargs,
    ):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz * self.n_passages, passage_length)
        if attention_mask is not None:
            attention_mask = attention_mask.view(bsz * self.n_passages,
                                                 passage_length)
        outputs = self.encoder(input_ids, attention_mask, *args, **kwargs)
        if isinstance(outputs, tuple):
            outputs = (outputs[0].view(bsz, self.n_passages * passage_length,
                                       -1), ) + outputs[1:]
        else:
            outputs.last_hidden_state = outputs.last_hidden_state.view(
                bsz, self.n_passages * passage_length, -1)
        return outputs


@MODELS.register_module(Tasks.fid_dialogue, module_name=Models.fid_T5)
class FIDT5Chat(T5Chat):
    """
    T5 model with FID(fuse-in-decoder) structure, mainly for dialogue tasks

    Parameters:
        model_dir: A path to a `directory` containing a configuration files to build model
    """

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.wrap_encoder()

    def wrap_encoder(self):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.backbone.encoder = EncoderWrapper(self.backbone.encoder)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.backbone.encoder = self.backbone.encoder.encoder
        block = []
        for mod in self.backbone.encoder.block:
            block.append(mod.module)
        block = torch.nn.ModuleList(block)
        self.backbone.encoder.block = block

    def load(self,
             pretrained_model_path,
             from_tf=False):  # only invoked when model is not onnx format
        self.unwrap_encoder()
        super().load(pretrained_model_path)
        self.wrap_encoder()

    def generate(self, inputs, *args, **kwargs) -> TokenGeneratorOutput:
        input_ids = inputs.get('input_ids')
        n_passages = input_ids.size(1)

        self.backbone.eval()
        with torch.no_grad():
            self.backbone.encoder.set_n_passages(n_passages)
            input_ids = input_ids.view(input_ids.size(0), -1)
            response = super().generate(input_ids, *args, **kwargs)
        return response

    def forward(self, input_ids, decoder_input_ids, *args, **kwargs):
        """
        The forward function of the model.

        Args:
           input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`
           or `(batch_size, n_passages, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
           decoder_input_ids (`torch.LongTensor` of shape `(batch_size,target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.
        Returns:
           Returns `modelscope.outputs.nlp_outputs.TextGenerationModelOutput`
        """
        attention_mask = input_ids.ne(0).long()
        if input_ids is not None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.backbone.encoder.set_n_passages(input_ids.size(1))
            input_ids = input_ids.view(input_ids.size(0), -1)
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        seq2seq_lm_output = self.backbone.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=decoder_input_ids,
            *args,
            **kwargs)
        return TextGenerationModelOutput(
            loss=seq2seq_lm_output.loss, logits=seq2seq_lm_output.logits)
