# Copyright (c) 2022 Zhipu.AI
"""Multiple choice model."""

import torch
import torch.nn

from .modeling_glm import GLMModel


class GLMForMultiTokenCloze(torch.nn.Module):

    def __init__(self,
                 language_model: GLMModel,
                 take_softmax=True,
                 length_penalty=0.0):
        super(GLMForMultiTokenCloze, self).__init__()
        self.model = language_model
        self.take_softmax = take_softmax
        self.length_penalty = length_penalty

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # [h.remove() for h in self.hook_handles]
        sd = self.model.state_dict(destination, prefix, keep_vars)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.model.named_parameters(prefix=prefix, recurse=recurse)

    def forward(self,
                input_ids,
                position_ids,
                attention_mask,
                target_ids=None,
                logit_mask=None,
                prompt_pos=None):
        if target_ids is None:
            return self.model(input_ids, position_ids, attention_mask)
        num_choices = None
        if len(input_ids.shape) == 3:
            batch_size, num_choices = input_ids.shape[:2]
            input_ids = input_ids.reshape(-1, input_ids.size(-1))
            attention_mask = attention_mask.reshape(-1,
                                                    *attention_mask.size()[2:])
            position_ids = position_ids.reshape(-1, *position_ids.size()[2:])
            target_ids = target_ids.reshape(-1, target_ids.size(-1))
            logit_mask = logit_mask.reshape(-1, logit_mask.size(-1))
            if prompt_pos is not None:
                prompt_pos = prompt_pos.reshape(-1, prompt_pos.size(-1))
        outputs, *mems = self.model(
            input_ids, position_ids, attention_mask, prompt_pos=prompt_pos)
        if self.take_softmax:
            outputs = torch.nn.functional.log_softmax(outputs, dim=-1)
        # select the target logits
        batch_ids = torch.arange(
            target_ids.size(0), dtype=torch.long, device=target_ids.device)
        batch_ids = batch_ids.unsqueeze(1).expand_as(target_ids)
        seq_ids = torch.arange(
            target_ids.size(-1), dtype=torch.long, device=target_ids.device)
        seq_ids = seq_ids.unsqueeze(0).expand_as(target_ids)
        logits = outputs[batch_ids, seq_ids, target_ids]
        logits = (logits * logit_mask).sum(dim=1)
        if self.length_penalty > 0.0:
            logits = logits / logit_mask.sum(dim=1)**self.length_penalty
        if num_choices is not None:
            logits = logits.view(-1, num_choices)
        return (logits, *mems)


class GLMForMultiTokenClozeFast(torch.nn.Module):

    def __init__(self, language_model, take_softmax=True, length_penalty=0.0):
        super(GLMForMultiTokenClozeFast, self).__init__()
        self.model = language_model
        self.take_softmax = take_softmax
        self.length_penalty = length_penalty

    def forward(self, input_ids, position_ids, attention_mask, dec_input_ids,
                dec_position_ids, dec_attention_mask, dec_target_ids,
                dec_logit_mask):
        # encoder
        outputs, *mems = self.model(
            input_ids,
            position_ids,
            attention_mask,
            return_memory=True,
            detach_memory=False)
        batch_size, num_choices, max_dec_len = dec_input_ids.size()
        max_enc_len = input_ids.size(-1)

        enc_mems = []
        for hidden in mems:
            hidden = hidden.unsqueeze(1).expand(-1, num_choices, -1,
                                                -1).reshape(
                                                    batch_size * num_choices,
                                                    *hidden.size()[1:])
            enc_mems.append(hidden)

        def build_dec_mask_matrix(seq_length, sep, memory_length=0):
            m = enc_mems[0].new_ones((1, seq_length, seq_length))
            m = torch.tril(m)

            # sep = dec_attention_mask
            ids = torch.arange(
                memory_length, device=sep.device, dtype=sep.dtype).view(1, -1)
            mask = ids < sep.view(-1, 1)  # batch * mem
            mask = mask.unsqueeze(1).float().expand(-1, seq_length, -1)

            m = m.expand(batch_size * num_choices, -1, -1)
            m = torch.cat((mask, m), dim=2)
            m = m.unsqueeze(1)
            return m

        dec_input_ids = dec_input_ids.reshape(-1, max_dec_len)
        dec_position_ids = dec_position_ids.reshape(
            -1,
            *dec_position_ids.size()[2:])
        # dec_attention_mask = dec_attention_mask.reshape(-1, *dec_attention_mask.size()[2:]).unsqueeze(1)
        dec_attention_mask = build_dec_mask_matrix(
            max_dec_len, dec_attention_mask.reshape(-1), max_enc_len)
        dec_target_ids = dec_target_ids.reshape(-1, dec_target_ids.size(-1))
        dec_logit_mask = dec_logit_mask.reshape(-1, dec_logit_mask.size(-1))

        outputs, *mems = self.model(dec_input_ids, dec_position_ids,
                                    dec_attention_mask, *enc_mems)
        if self.take_softmax:
            outputs = torch.nn.functional.log_softmax(outputs, dim=-1)

        batch_ids = torch.arange(
            dec_target_ids.size(0),
            dtype=torch.long,
            device=dec_target_ids.device)
        batch_ids = batch_ids.unsqueeze(1).expand_as(dec_target_ids)
        seq_ids = torch.arange(
            dec_target_ids.size(-1),
            dtype=torch.long,
            device=dec_target_ids.device)
        seq_ids = seq_ids.unsqueeze(0).expand_as(dec_target_ids)
        logits = outputs[batch_ids, seq_ids, dec_target_ids]
        logits = (logits * dec_logit_mask).sum(dim=1)
        if self.length_penalty > 0.0:
            logits = logits / dec_logit_mask.sum(dim=1)**self.length_penalty
        if num_choices is not None:
            logits = logits.view(-1, num_choices)
        return (logits, *mems)


class GLMForSingleTokenCloze(torch.nn.Module):

    def __init__(self, language_model, take_softmax=False):
        super().__init__()
        self.model = language_model
        self.take_softmax = take_softmax

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # [h.remove() for h in self.hook_handles]
        sd = self.model.state_dict(destination, prefix, keep_vars)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.model.named_parameters(prefix=prefix, recurse=recurse)

    def forward(self,
                input_ids,
                position_ids,
                attention_mask,
                target_ids=None,
                logit_mask=None,
                prompt_pos=None):
        if target_ids is None:
            return self.model(input_ids, position_ids, attention_mask)
        assert len(input_ids.shape) == 2
        outputs, *mems = self.model(
            input_ids, position_ids, attention_mask, prompt_pos=prompt_pos)
        batch_ids = torch.arange(
            outputs.size(0),
            dtype=attention_mask.dtype,
            device=attention_mask.device)
        target_logits = outputs[batch_ids, attention_mask]
        if self.take_softmax:
            target_prob = torch.nn.functional.log_softmax(
                target_logits, dim=-1)
        else:
            target_prob = target_logits
        batch_ids = batch_ids.unsqueeze(1).expand_as(target_ids)
        output = target_prob[batch_ids, target_ids]

        return (output, target_logits, *mems)


class GLMForSequenceClassification(torch.nn.Module):

    def __init__(self,
                 language_model,
                 hidden_size,
                 hidden_dropout,
                 pool_token,
                 num_class=1):
        super().__init__()
        self.pool_token = pool_token
        self.model = language_model
        self.num_class = num_class
        # Multi-choice head.
        self.pool_layer = torch.nn.Linear(hidden_size, hidden_size)
        self.multichoice_dropout = torch.nn.Dropout(hidden_dropout)
        self.multichoice_head = torch.nn.Linear(hidden_size, num_class)

    def forward(self, input_ids, position_ids, attention_mask):
        num_choices = None
        if len(input_ids.shape) == 3:
            assert self.num_class == 1
            batch_size, num_choices = input_ids.shape[:2]
            input_ids = input_ids.reshape(-1, input_ids.size(-1))
            attention_mask = attention_mask.reshape(-1,
                                                    *attention_mask.size()[2:])
            position_ids = position_ids.reshape(-1, *position_ids.size()[2:])
        outputs, *mems = self.model(input_ids, position_ids, attention_mask)
        if self.pool_token == 'start':
            output = outputs[torch.arange(
                outputs.size(0),
                dtype=attention_mask.dtype,
                device=attention_mask.device), attention_mask]
        elif self.pool_token == 'pad':
            output = outputs[torch.arange(
                outputs.size(0),
                dtype=attention_mask.dtype,
                device=attention_mask.device), attention_mask - 1]
        elif self.pool_token == 'cls':
            output = outputs[:, 0]
        else:
            raise NotImplementedError
        output = torch.tanh(self.pool_layer(output))
        multichoice_output = self.multichoice_dropout(output)
        logits = self.multichoice_head(multichoice_output)
        if num_choices is not None:
            logits = logits.view(-1, num_choices)
        return (logits, *mems)
