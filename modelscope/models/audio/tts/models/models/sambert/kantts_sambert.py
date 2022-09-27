# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.models.audio.tts.models.utils import get_mask_from_lengths
from .adaptors import (LengthRegulator, VarFsmnRnnNARPredictor,
                       VarRnnARPredictor)
from .base import FFTBlock, PNCABlock, Prenet
from .fsmn import FsmnEncoderV2
from .positions import DurSinusoidalPositionEncoder, SinusoidalPositionEncoder


class SelfAttentionEncoder(nn.Module):

    def __init__(self, n_layer, d_in, d_model, n_head, d_head, d_inner,
                 dropout, dropout_att, dropout_relu, position_encoder):
        super(SelfAttentionEncoder, self).__init__()

        self.d_in = d_in
        self.d_model = d_model
        self.dropout = dropout
        d_in_lst = [d_in] + [d_model] * (n_layer - 1)
        self.fft = nn.ModuleList([
            FFTBlock(d, d_model, n_head, d_head, d_inner, (3, 1), dropout,
                     dropout_att, dropout_relu) for d in d_in_lst
        ])
        self.ln = nn.LayerNorm(d_model, eps=1e-6)
        self.position_enc = position_encoder

    def forward(self, input, mask=None, return_attns=False):
        input *= self.d_model**0.5
        if (isinstance(self.position_enc, SinusoidalPositionEncoder)):
            input = self.position_enc(input)
        else:
            raise NotImplementedError('modelscope error: position_enc invalid')

        input = F.dropout(input, p=self.dropout, training=self.training)

        enc_slf_attn_list = []
        max_len = input.size(1)
        if mask is not None:
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        else:
            slf_attn_mask = None

        enc_output = input
        for id, layer in enumerate(self.fft):
            enc_output, enc_slf_attn = layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        enc_output = self.ln(enc_output)

        return enc_output, enc_slf_attn_list


class HybridAttentionDecoder(nn.Module):

    def __init__(self, d_in, prenet_units, n_layer, d_model, d_mem, n_head,
                 d_head, d_inner, dropout, dropout_att, dropout_relu, d_out):
        super(HybridAttentionDecoder, self).__init__()

        self.d_model = d_model
        self.dropout = dropout
        self.prenet = Prenet(d_in, prenet_units, d_model)
        self.dec_in_proj = nn.Linear(d_model + d_mem, d_model)
        self.pnca = nn.ModuleList([
            PNCABlock(d_model, d_mem, n_head, d_head, d_inner, (1, 1), dropout,
                      dropout_att, dropout_relu) for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(d_model, eps=1e-6)
        self.dec_out_proj = nn.Linear(d_model, d_out)

    def reset_state(self):
        for layer in self.pnca:
            layer.reset_state()

    def get_pnca_attn_mask(self,
                           device,
                           max_len,
                           x_band_width,
                           h_band_width,
                           mask=None):
        if mask is not None:
            pnca_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        else:
            pnca_attn_mask = None

        range_ = torch.arange(max_len).to(device)
        x_start = torch.clamp_min(range_ - x_band_width, 0)[None, None, :]
        x_end = (range_ + 1)[None, None, :]
        h_start = range_[None, None, :]
        h_end = torch.clamp_max(range_ + h_band_width + 1,
                                max_len + 1)[None, None, :]

        pnca_x_attn_mask = ~((x_start <= range_[None, :, None])
                             & (x_end > range_[None, :, None])).transpose(1, 2)  # yapf:disable
        pnca_h_attn_mask = ~((h_start <= range_[None, :, None])
                             & (h_end > range_[None, :, None])).transpose(1, 2)  # yapf:disable

        if pnca_attn_mask is not None:
            pnca_x_attn_mask = (pnca_x_attn_mask | pnca_attn_mask)
            pnca_h_attn_mask = (pnca_h_attn_mask | pnca_attn_mask)
            pnca_x_attn_mask = pnca_x_attn_mask.masked_fill(
                pnca_attn_mask.transpose(1, 2), False)
            pnca_h_attn_mask = pnca_h_attn_mask.masked_fill(
                pnca_attn_mask.transpose(1, 2), False)

        return pnca_attn_mask, pnca_x_attn_mask, pnca_h_attn_mask

    # must call reset_state before
    def forward(self,
                input,
                memory,
                x_band_width,
                h_band_width,
                mask=None,
                return_attns=False):
        input = self.prenet(input)
        input = torch.cat([memory, input], dim=-1)
        input = self.dec_in_proj(input)

        if mask is not None:
            input = input.masked_fill(mask.unsqueeze(-1), 0)

        input *= self.d_model**0.5
        input = F.dropout(input, p=self.dropout, training=self.training)

        max_len = input.size(1)
        pnca_attn_mask, pnca_x_attn_mask, pnca_h_attn_mask = self.get_pnca_attn_mask(
            input.device, max_len, x_band_width, h_band_width, mask)

        dec_pnca_attn_x_list = []
        dec_pnca_attn_h_list = []
        dec_output = input
        for id, layer in enumerate(self.pnca):
            dec_output, dec_pnca_attn_x, dec_pnca_attn_h = layer(
                dec_output,
                memory,
                mask=mask,
                pnca_x_attn_mask=pnca_x_attn_mask,
                pnca_h_attn_mask=pnca_h_attn_mask)
            if return_attns:
                dec_pnca_attn_x_list += [dec_pnca_attn_x]
                dec_pnca_attn_h_list += [dec_pnca_attn_h]

        dec_output = self.ln(dec_output)
        dec_output = self.dec_out_proj(dec_output)

        return dec_output, dec_pnca_attn_x_list, dec_pnca_attn_h_list

    # must call reset_state before when step == 0
    def infer(self,
              step,
              input,
              memory,
              x_band_width,
              h_band_width,
              mask=None,
              return_attns=False):
        max_len = memory.size(1)

        input = self.prenet(input)
        input = torch.cat([memory[:, step:step + 1, :], input], dim=-1)
        input = self.dec_in_proj(input)

        input *= self.d_model**0.5
        input = F.dropout(input, p=self.dropout, training=self.training)

        pnca_attn_mask, pnca_x_attn_mask, pnca_h_attn_mask = self.get_pnca_attn_mask(
            input.device, max_len, x_band_width, h_band_width, mask)

        dec_pnca_attn_x_list = []
        dec_pnca_attn_h_list = []
        dec_output = input
        for id, layer in enumerate(self.pnca):
            if mask is not None:
                mask_step = mask[:, step:step + 1]
            else:
                mask_step = None
            dec_output, dec_pnca_attn_x, dec_pnca_attn_h = layer(
                dec_output,
                memory,
                mask=mask_step,
                pnca_x_attn_mask=pnca_x_attn_mask[:,
                                                  step:step + 1, :(step + 1)],
                pnca_h_attn_mask=pnca_h_attn_mask[:, step:step + 1, :])
            if return_attns:
                dec_pnca_attn_x_list += [dec_pnca_attn_x]
                dec_pnca_attn_h_list += [dec_pnca_attn_h]

        dec_output = self.ln(dec_output)
        dec_output = self.dec_out_proj(dec_output)

        return dec_output, dec_pnca_attn_x_list, dec_pnca_attn_h_list


class TextFftEncoder(nn.Module):

    def __init__(self, config, ling_unit_size):
        super(TextFftEncoder, self).__init__()

        # linguistic unit lookup table
        nb_ling_sy = ling_unit_size['sy']
        nb_ling_tone = ling_unit_size['tone']
        nb_ling_syllable_flag = ling_unit_size['syllable_flag']
        nb_ling_ws = ling_unit_size['word_segment']

        max_len = config['am']['max_len']

        d_emb = config['am']['embedding_dim']
        nb_layers = config['am']['encoder_num_layers']
        nb_heads = config['am']['encoder_num_heads']
        d_model = config['am']['encoder_num_units']
        d_head = d_model // nb_heads
        d_inner = config['am']['encoder_ffn_inner_dim']
        dropout = config['am']['encoder_dropout']
        dropout_attn = config['am']['encoder_attention_dropout']
        dropout_relu = config['am']['encoder_relu_dropout']
        d_proj = config['am']['encoder_projection_units']

        self.d_model = d_model

        self.sy_emb = nn.Embedding(nb_ling_sy, d_emb)
        self.tone_emb = nn.Embedding(nb_ling_tone, d_emb)
        self.syllable_flag_emb = nn.Embedding(nb_ling_syllable_flag, d_emb)
        self.ws_emb = nn.Embedding(nb_ling_ws, d_emb)

        position_enc = SinusoidalPositionEncoder(max_len, d_emb)

        self.ling_enc = SelfAttentionEncoder(nb_layers, d_emb, d_model,
                                             nb_heads, d_head, d_inner,
                                             dropout, dropout_attn,
                                             dropout_relu, position_enc)

        self.ling_proj = nn.Linear(d_model, d_proj, bias=False)

    def forward(self, inputs_ling, masks=None, return_attns=False):
        # Parse inputs_ling_seq
        inputs_sy = inputs_ling[:, :, 0]
        inputs_tone = inputs_ling[:, :, 1]
        inputs_syllable_flag = inputs_ling[:, :, 2]
        inputs_ws = inputs_ling[:, :, 3]

        # Lookup table
        sy_embedding = self.sy_emb(inputs_sy)
        tone_embedding = self.tone_emb(inputs_tone)
        syllable_flag_embedding = self.syllable_flag_emb(inputs_syllable_flag)
        ws_embedding = self.ws_emb(inputs_ws)

        ling_embedding = sy_embedding + tone_embedding + syllable_flag_embedding + ws_embedding

        enc_output, enc_slf_attn_list = self.ling_enc(ling_embedding, masks,
                                                      return_attns)

        enc_output = self.ling_proj(enc_output)

        return enc_output, enc_slf_attn_list


class VarianceAdaptor(nn.Module):

    def __init__(self, config):
        super(VarianceAdaptor, self).__init__()

        input_dim = config['am']['encoder_projection_units'] + config['am'][
            'emotion_units'] + config['am']['speaker_units']
        filter_size = config['am']['predictor_filter_size']
        fsmn_num_layers = config['am']['predictor_fsmn_num_layers']
        num_memory_units = config['am']['predictor_num_memory_units']
        ffn_inner_dim = config['am']['predictor_ffn_inner_dim']
        dropout = config['am']['predictor_dropout']
        shift = config['am']['predictor_shift']
        lstm_units = config['am']['predictor_lstm_units']

        dur_pred_prenet_units = config['am']['dur_pred_prenet_units']
        dur_pred_lstm_units = config['am']['dur_pred_lstm_units']

        self.pitch_predictor = VarFsmnRnnNARPredictor(input_dim, filter_size,
                                                      fsmn_num_layers,
                                                      num_memory_units,
                                                      ffn_inner_dim, dropout,
                                                      shift, lstm_units)
        self.energy_predictor = VarFsmnRnnNARPredictor(input_dim, filter_size,
                                                       fsmn_num_layers,
                                                       num_memory_units,
                                                       ffn_inner_dim, dropout,
                                                       shift, lstm_units)
        self.duration_predictor = VarRnnARPredictor(input_dim,
                                                    dur_pred_prenet_units,
                                                    dur_pred_lstm_units)

        self.length_regulator = LengthRegulator(
            config['am']['outputs_per_step'])
        self.dur_position_encoder = DurSinusoidalPositionEncoder(
            config['am']['encoder_projection_units'],
            config['am']['outputs_per_step'])

        self.pitch_emb = nn.Conv1d(
            1,
            config['am']['encoder_projection_units'],
            kernel_size=9,
            padding=4)
        self.energy_emb = nn.Conv1d(
            1,
            config['am']['encoder_projection_units'],
            kernel_size=9,
            padding=4)

    def forward(self,
                inputs_text_embedding,
                inputs_emo_embedding,
                inputs_spk_embedding,
                masks=None,
                output_masks=None,
                duration_targets=None,
                pitch_targets=None,
                energy_targets=None):

        batch_size = inputs_text_embedding.size(0)

        variance_predictor_inputs = torch.cat([
            inputs_text_embedding, inputs_spk_embedding, inputs_emo_embedding
        ], dim=-1)  # yapf:disable

        pitch_predictions = self.pitch_predictor(variance_predictor_inputs,
                                                 masks)
        energy_predictions = self.energy_predictor(variance_predictor_inputs,
                                                   masks)

        if pitch_targets is not None:
            pitch_embeddings = self.pitch_emb(
                pitch_targets.unsqueeze(1)).transpose(1, 2)
        else:
            pitch_embeddings = self.pitch_emb(
                pitch_predictions.unsqueeze(1)).transpose(1, 2)

        if energy_targets is not None:
            energy_embeddings = self.energy_emb(
                energy_targets.unsqueeze(1)).transpose(1, 2)
        else:
            energy_embeddings = self.energy_emb(
                energy_predictions.unsqueeze(1)).transpose(1, 2)

        inputs_text_embedding_aug = inputs_text_embedding + pitch_embeddings + energy_embeddings
        duration_predictor_cond = torch.cat([
            inputs_text_embedding_aug, inputs_spk_embedding,
            inputs_emo_embedding
        ], dim=-1)  # yapf:disable
        if duration_targets is not None:
            duration_predictor_go_frame = torch.zeros(batch_size, 1).to(
                inputs_text_embedding.device)
            duration_predictor_input = torch.cat([
                duration_predictor_go_frame, duration_targets[:, :-1].float()
            ], dim=-1)  # yapf:disable
            duration_predictor_input = torch.log(duration_predictor_input + 1)
            log_duration_predictions, _ = self.duration_predictor(
                duration_predictor_input.unsqueeze(-1),
                duration_predictor_cond,
                masks=masks)
            duration_predictions = torch.exp(log_duration_predictions) - 1
        else:
            log_duration_predictions = self.duration_predictor.infer(
                duration_predictor_cond, masks=masks)
            duration_predictions = torch.exp(log_duration_predictions) - 1

        if duration_targets is not None:
            LR_text_outputs, LR_length_rounded = self.length_regulator(
                inputs_text_embedding_aug,
                duration_targets,
                masks=output_masks)
            LR_position_embeddings = self.dur_position_encoder(
                duration_targets, masks=output_masks)
            LR_emo_outputs, _ = self.length_regulator(
                inputs_emo_embedding, duration_targets, masks=output_masks)
            LR_spk_outputs, _ = self.length_regulator(
                inputs_spk_embedding, duration_targets, masks=output_masks)

        else:
            LR_text_outputs, LR_length_rounded = self.length_regulator(
                inputs_text_embedding_aug,
                duration_predictions,
                masks=output_masks)
            LR_position_embeddings = self.dur_position_encoder(
                duration_predictions, masks=output_masks)
            LR_emo_outputs, _ = self.length_regulator(
                inputs_emo_embedding, duration_predictions, masks=output_masks)
            LR_spk_outputs, _ = self.length_regulator(
                inputs_spk_embedding, duration_predictions, masks=output_masks)

        LR_text_outputs = LR_text_outputs + LR_position_embeddings

        return (LR_text_outputs, LR_emo_outputs, LR_spk_outputs,
                LR_length_rounded, log_duration_predictions, pitch_predictions,
                energy_predictions)


class MelPNCADecoder(nn.Module):

    def __init__(self, config):
        super(MelPNCADecoder, self).__init__()

        prenet_units = config['am']['decoder_prenet_units']
        nb_layers = config['am']['decoder_num_layers']
        nb_heads = config['am']['decoder_num_heads']
        d_model = config['am']['decoder_num_units']
        d_head = d_model // nb_heads
        d_inner = config['am']['decoder_ffn_inner_dim']
        dropout = config['am']['decoder_dropout']
        dropout_attn = config['am']['decoder_attention_dropout']
        dropout_relu = config['am']['decoder_relu_dropout']
        outputs_per_step = config['am']['outputs_per_step']

        d_mem = config['am'][
            'encoder_projection_units'] * outputs_per_step + config['am'][
                'emotion_units'] + config['am']['speaker_units']
        d_mel = config['am']['num_mels']

        self.d_mel = d_mel
        self.r = outputs_per_step
        self.nb_layers = nb_layers

        self.mel_dec = HybridAttentionDecoder(d_mel, prenet_units, nb_layers,
                                              d_model, d_mem, nb_heads, d_head,
                                              d_inner, dropout, dropout_attn,
                                              dropout_relu,
                                              d_mel * outputs_per_step)

    def forward(self,
                memory,
                x_band_width,
                h_band_width,
                target=None,
                mask=None,
                return_attns=False):
        batch_size = memory.size(0)
        go_frame = torch.zeros((batch_size, 1, self.d_mel)).to(memory.device)

        if target is not None:
            self.mel_dec.reset_state()
            input = target[:, self.r - 1::self.r, :]
            input = torch.cat([go_frame, input], dim=1)[:, :-1, :]
            dec_output, dec_pnca_attn_x_list, dec_pnca_attn_h_list = self.mel_dec(
                input,
                memory,
                x_band_width,
                h_band_width,
                mask=mask,
                return_attns=return_attns)

        else:
            dec_output = []
            dec_pnca_attn_x_list = [[] for _ in range(self.nb_layers)]
            dec_pnca_attn_h_list = [[] for _ in range(self.nb_layers)]
            self.mel_dec.reset_state()
            input = go_frame
            for step in range(memory.size(1)):
                dec_output_step, dec_pnca_attn_x_step, dec_pnca_attn_h_step = self.mel_dec.infer(
                    step,
                    input,
                    memory,
                    x_band_width,
                    h_band_width,
                    mask=mask,
                    return_attns=return_attns)
                input = dec_output_step[:, :, -self.d_mel:]

                dec_output.append(dec_output_step)
                for layer_id, (pnca_x_attn, pnca_h_attn) in enumerate(
                        zip(dec_pnca_attn_x_step, dec_pnca_attn_h_step)):
                    left = memory.size(1) - pnca_x_attn.size(-1)
                    if (left > 0):
                        padding = torch.zeros(
                            (pnca_x_attn.size(0), 1, left)).to(pnca_x_attn)
                        pnca_x_attn = torch.cat([pnca_x_attn, padding], dim=-1)
                    dec_pnca_attn_x_list[layer_id].append(pnca_x_attn)
                    dec_pnca_attn_h_list[layer_id].append(pnca_h_attn)

            dec_output = torch.cat(dec_output, dim=1)
            for layer_id in range(self.nb_layers):
                dec_pnca_attn_x_list[layer_id] = torch.cat(
                    dec_pnca_attn_x_list[layer_id], dim=1)
                dec_pnca_attn_h_list[layer_id] = torch.cat(
                    dec_pnca_attn_h_list[layer_id], dim=1)

        return dec_output, dec_pnca_attn_x_list, dec_pnca_attn_h_list


class PostNet(nn.Module):

    def __init__(self, config):
        super(PostNet, self).__init__()

        self.filter_size = config['am']['postnet_filter_size']
        self.fsmn_num_layers = config['am']['postnet_fsmn_num_layers']
        self.num_memory_units = config['am']['postnet_num_memory_units']
        self.ffn_inner_dim = config['am']['postnet_ffn_inner_dim']
        self.dropout = config['am']['postnet_dropout']
        self.shift = config['am']['postnet_shift']
        self.lstm_units = config['am']['postnet_lstm_units']
        self.num_mels = config['am']['num_mels']

        self.fsmn = FsmnEncoderV2(self.filter_size, self.fsmn_num_layers,
                                  self.num_mels, self.num_memory_units,
                                  self.ffn_inner_dim, self.dropout, self.shift)
        self.lstm = nn.LSTM(
            self.num_memory_units,
            self.lstm_units,
            num_layers=1,
            batch_first=True)
        self.fc = nn.Linear(self.lstm_units, self.num_mels)

    def forward(self, x, mask=None):
        postnet_fsmn_output = self.fsmn(x, mask)
        # The input can also be a packed variable length sequence,
        # here we just omit it for simpliciy due to the mask and uni-directional lstm.
        postnet_lstm_output, _ = self.lstm(postnet_fsmn_output)
        mel_residual_output = self.fc(postnet_lstm_output)

        return mel_residual_output


def mel_recon_loss_fn(output_lengths,
                      mel_targets,
                      dec_outputs,
                      postnet_outputs=None):
    mae_loss = nn.L1Loss(reduction='none')

    output_masks = get_mask_from_lengths(
        output_lengths, max_len=mel_targets.size(1))
    output_masks = ~output_masks
    valid_outputs = output_masks.sum()

    mel_loss_ = torch.sum(
        mae_loss(mel_targets, dec_outputs) * output_masks.unsqueeze(-1)) / (
            valid_outputs * mel_targets.size(-1))

    if postnet_outputs is not None:
        mel_loss = torch.sum(
            mae_loss(mel_targets, postnet_outputs)
            * output_masks.unsqueeze(-1)) / (
                valid_outputs * mel_targets.size(-1))
    else:
        mel_loss = 0.0

    return mel_loss_, mel_loss


def prosody_recon_loss_fn(input_lengths, duration_targets, pitch_targets,
                          energy_targets, log_duration_predictions,
                          pitch_predictions, energy_predictions):
    mae_loss = nn.L1Loss(reduction='none')

    input_masks = get_mask_from_lengths(
        input_lengths, max_len=duration_targets.size(1))
    input_masks = ~input_masks
    valid_inputs = input_masks.sum()

    dur_loss = torch.sum(
        mae_loss(
            torch.log(duration_targets.float() + 1), log_duration_predictions)
        * input_masks) / valid_inputs
    pitch_loss = torch.sum(
        mae_loss(pitch_targets, pitch_predictions)
        * input_masks) / valid_inputs
    energy_loss = torch.sum(
        mae_loss(energy_targets, energy_predictions)
        * input_masks) / valid_inputs

    return dur_loss, pitch_loss, energy_loss


class KanTtsSAMBERT(nn.Module):

    def __init__(self, config, ling_unit_size):
        super(KanTtsSAMBERT, self).__init__()

        self.text_encoder = TextFftEncoder(config, ling_unit_size)
        self.spk_tokenizer = nn.Embedding(ling_unit_size['speaker'],
                                          config['am']['speaker_units'])
        self.emo_tokenizer = nn.Embedding(ling_unit_size['emotion'],
                                          config['am']['emotion_units'])
        self.variance_adaptor = VarianceAdaptor(config)
        self.mel_decoder = MelPNCADecoder(config)
        self.mel_postnet = PostNet(config)

    def get_lfr_mask_from_lengths(self, lengths, max_len):
        batch_size = lengths.size(0)
        # padding according to the outputs_per_step
        padded_lr_lengths = torch.zeros_like(lengths)
        for i in range(batch_size):
            len_item = int(lengths[i].item())
            padding = self.mel_decoder.r - len_item % self.mel_decoder.r
            if (padding < self.mel_decoder.r):
                padded_lr_lengths[i] = (len_item
                                        + padding) // self.mel_decoder.r
            else:
                padded_lr_lengths[i] = len_item // self.mel_decoder.r

        return get_mask_from_lengths(
            padded_lr_lengths, max_len=max_len // self.mel_decoder.r)

    def forward(self,
                inputs_ling,
                inputs_emotion,
                inputs_speaker,
                input_lengths,
                output_lengths=None,
                mel_targets=None,
                duration_targets=None,
                pitch_targets=None,
                energy_targets=None):

        batch_size = inputs_ling.size(0)

        input_masks = get_mask_from_lengths(
            input_lengths, max_len=inputs_ling.size(1))

        text_hid, enc_sla_attn_lst = self.text_encoder(
            inputs_ling, input_masks, return_attns=True)

        emo_hid = self.emo_tokenizer(inputs_emotion)
        spk_hid = self.spk_tokenizer(inputs_speaker)

        if output_lengths is not None:
            output_masks = get_mask_from_lengths(
                output_lengths, max_len=mel_targets.size(1))
        else:
            output_masks = None

        (LR_text_outputs, LR_emo_outputs, LR_spk_outputs, LR_length_rounded,
         log_duration_predictions, pitch_predictions,
         energy_predictions) = self.variance_adaptor(
             text_hid,
             emo_hid,
             spk_hid,
             masks=input_masks,
             output_masks=output_masks,
             duration_targets=duration_targets,
             pitch_targets=pitch_targets,
             energy_targets=energy_targets)

        if output_lengths is not None:
            lfr_masks = self.get_lfr_mask_from_lengths(
                output_lengths, max_len=LR_text_outputs.size(1))
        else:
            output_masks = get_mask_from_lengths(
                LR_length_rounded, max_len=LR_text_outputs.size(1))
            lfr_masks = None

        # LFR with the factor of outputs_per_step
        LFR_text_inputs = LR_text_outputs.contiguous().view(
            batch_size, -1, self.mel_decoder.r * text_hid.shape[-1])
        LFR_emo_inputs = LR_emo_outputs.contiguous().view(
            batch_size, -1,
            self.mel_decoder.r * emo_hid.shape[-1])[:, :, :emo_hid.shape[-1]]
        LFR_spk_inputs = LR_spk_outputs.contiguous().view(
            batch_size, -1,
            self.mel_decoder.r * spk_hid.shape[-1])[:, :, :spk_hid.shape[-1]]

        memory = torch.cat([LFR_text_inputs, LFR_spk_inputs, LFR_emo_inputs],
                           dim=-1)

        if duration_targets is not None:
            x_band_width = int(
                duration_targets.float().masked_fill(input_masks, 0).max()
                / self.mel_decoder.r + 0.5)
            h_band_width = x_band_width
        else:
            x_band_width = int((torch.exp(log_duration_predictions) - 1).max()
                               / self.mel_decoder.r + 0.5)
            h_band_width = x_band_width

        dec_outputs, pnca_x_attn_lst, pnca_h_attn_lst = self.mel_decoder(
            memory,
            x_band_width,
            h_band_width,
            target=mel_targets,
            mask=lfr_masks,
            return_attns=True)

        # De-LFR with the factor of outputs_per_step
        dec_outputs = dec_outputs.contiguous().view(batch_size, -1,
                                                    self.mel_decoder.d_mel)

        if output_masks is not None:
            dec_outputs = dec_outputs.masked_fill(
                output_masks.unsqueeze(-1), 0)

        postnet_outputs = self.mel_postnet(dec_outputs,
                                           output_masks) + dec_outputs
        if output_masks is not None:
            postnet_outputs = postnet_outputs.masked_fill(
                output_masks.unsqueeze(-1), 0)

        res = {
            'x_band_width': x_band_width,
            'h_band_width': h_band_width,
            'enc_slf_attn_lst': enc_sla_attn_lst,
            'pnca_x_attn_lst': pnca_x_attn_lst,
            'pnca_h_attn_lst': pnca_h_attn_lst,
            'dec_outputs': dec_outputs,
            'postnet_outputs': postnet_outputs,
            'LR_length_rounded': LR_length_rounded,
            'log_duration_predictions': log_duration_predictions,
            'pitch_predictions': pitch_predictions,
            'energy_predictions': energy_predictions
        }

        res['LR_text_outputs'] = LR_text_outputs
        res['LR_emo_outputs'] = LR_emo_outputs
        res['LR_spk_outputs'] = LR_spk_outputs

        return res
