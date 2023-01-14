# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.models.audio.tts.kantts.models.utils import \
    get_mask_from_lengths
from . import FFTBlock, PNCABlock, Prenet
from .adaptors import (LengthRegulator, VarFsmnRnnNARPredictor,
                       VarRnnARPredictor)
from .alignment import b_mas
from .attention import ConvAttention
from .fsmn import FsmnEncoderV2
from .positions import DurSinusoidalPositionEncoder, SinusoidalPositionEncoder


class SelfAttentionEncoder(nn.Module):

    def __init__(
        self,
        n_layer,
        d_in,
        d_model,
        n_head,
        d_head,
        d_inner,
        dropout,
        dropout_att,
        dropout_relu,
        position_encoder,
    ):
        super(SelfAttentionEncoder, self).__init__()

        self.d_in = d_in
        self.d_model = d_model
        self.dropout = dropout
        d_in_lst = [d_in] + [d_model] * (n_layer - 1)
        self.fft = nn.ModuleList([
            FFTBlock(
                d,
                d_model,
                n_head,
                d_head,
                d_inner,
                (3, 1),
                dropout,
                dropout_att,
                dropout_relu,
            ) for d in d_in_lst
        ])
        self.ln = nn.LayerNorm(d_model, eps=1e-6)
        self.position_enc = position_encoder

    def forward(self, input, mask=None, return_attns=False):
        input *= self.d_model**0.5
        if isinstance(self.position_enc, SinusoidalPositionEncoder):
            input = self.position_enc(input)
        else:
            raise NotImplementedError

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

    def __init__(
        self,
        d_in,
        prenet_units,
        n_layer,
        d_model,
        d_mem,
        n_head,
        d_head,
        d_inner,
        dropout,
        dropout_att,
        dropout_relu,
        d_out,
    ):
        super(HybridAttentionDecoder, self).__init__()

        self.d_model = d_model
        self.dropout = dropout
        self.prenet = Prenet(d_in, prenet_units, d_model)
        self.dec_in_proj = nn.Linear(d_model + d_mem, d_model)
        self.pnca = nn.ModuleList([
            PNCABlock(
                d_model,
                d_mem,
                n_head,
                d_head,
                d_inner,
                (1, 1),
                dropout,
                dropout_att,
                dropout_relu,
            ) for _ in range(n_layer)
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
                             &  # noqa W504
                             (x_end > range_[None, :, None])).transpose(1, 2)
        pnca_h_attn_mask = ~((h_start <= range_[None, :, None])
                             &  # noqa W504
                             (h_end > range_[None, :, None])).transpose(1, 2)

        if pnca_attn_mask is not None:
            pnca_x_attn_mask = pnca_x_attn_mask | pnca_attn_mask
            pnca_h_attn_mask = pnca_h_attn_mask | pnca_attn_mask
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
                pnca_h_attn_mask=pnca_h_attn_mask,
            )
            if return_attns:
                dec_pnca_attn_x_list += [dec_pnca_attn_x]
                dec_pnca_attn_h_list += [dec_pnca_attn_h]

        dec_output = self.ln(dec_output)
        dec_output = self.dec_out_proj(dec_output)

        return dec_output, dec_pnca_attn_x_list, dec_pnca_attn_h_list

    # must call reset_state before when step == 0
    def infer(
        self,
        step,
        input,
        memory,
        x_band_width,
        h_band_width,
        mask=None,
        return_attns=False,
    ):
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
                pnca_h_attn_mask=pnca_h_attn_mask[:, step:step + 1, :],
            )
            if return_attns:
                dec_pnca_attn_x_list += [dec_pnca_attn_x]
                dec_pnca_attn_h_list += [dec_pnca_attn_h]

        dec_output = self.ln(dec_output)
        dec_output = self.dec_out_proj(dec_output)

        return dec_output, dec_pnca_attn_x_list, dec_pnca_attn_h_list


class TextFftEncoder(nn.Module):

    def __init__(self, config):
        super(TextFftEncoder, self).__init__()

        d_emb = config['embedding_dim']
        self.using_byte = False
        if config.get('using_byte', False):
            self.using_byte = True
            nb_ling_byte_index = config['byte_index']
            self.byte_index_emb = nn.Embedding(nb_ling_byte_index, d_emb)
        else:
            # linguistic unit lookup table
            nb_ling_sy = config['sy']
            nb_ling_tone = config['tone']
            nb_ling_syllable_flag = config['syllable_flag']
            nb_ling_ws = config['word_segment']
            self.sy_emb = nn.Embedding(nb_ling_sy, d_emb)
            self.tone_emb = nn.Embedding(nb_ling_tone, d_emb)
            self.syllable_flag_emb = nn.Embedding(nb_ling_syllable_flag, d_emb)
            self.ws_emb = nn.Embedding(nb_ling_ws, d_emb)

        max_len = config['max_len']

        nb_layers = config['encoder_num_layers']
        nb_heads = config['encoder_num_heads']
        d_model = config['encoder_num_units']
        d_head = d_model // nb_heads
        d_inner = config['encoder_ffn_inner_dim']
        dropout = config['encoder_dropout']
        dropout_attn = config['encoder_attention_dropout']
        dropout_relu = config['encoder_relu_dropout']
        d_proj = config['encoder_projection_units']

        self.d_model = d_model

        position_enc = SinusoidalPositionEncoder(max_len, d_emb)

        self.ling_enc = SelfAttentionEncoder(
            nb_layers,
            d_emb,
            d_model,
            nb_heads,
            d_head,
            d_inner,
            dropout,
            dropout_attn,
            dropout_relu,
            position_enc,
        )

        self.ling_proj = nn.Linear(d_model, d_proj, bias=False)

    def forward(self, inputs_ling, masks=None, return_attns=False):
        # Parse inputs_ling_seq
        if self.using_byte:
            inputs_byte_index = inputs_ling[:, :, 0]
            byte_index_embedding = self.byte_index_emb(inputs_byte_index)
            ling_embedding = byte_index_embedding
        else:
            inputs_sy = inputs_ling[:, :, 0]
            inputs_tone = inputs_ling[:, :, 1]
            inputs_syllable_flag = inputs_ling[:, :, 2]
            inputs_ws = inputs_ling[:, :, 3]

            # Lookup table
            sy_embedding = self.sy_emb(inputs_sy)
            tone_embedding = self.tone_emb(inputs_tone)
            syllable_flag_embedding = self.syllable_flag_emb(
                inputs_syllable_flag)
            ws_embedding = self.ws_emb(inputs_ws)

            ling_embedding = (
                sy_embedding + tone_embedding + syllable_flag_embedding
                + ws_embedding)

        enc_output, enc_slf_attn_list = self.ling_enc(ling_embedding, masks,
                                                      return_attns)

        if hasattr(self, 'ling_proj'):
            enc_output = self.ling_proj(enc_output)

        return enc_output, enc_slf_attn_list, ling_embedding


class VarianceAdaptor(nn.Module):

    def __init__(self, config):
        super(VarianceAdaptor, self).__init__()

        input_dim = (
            config['encoder_projection_units'] + config['emotion_units']
            + config['speaker_units'])
        filter_size = config['predictor_filter_size']
        fsmn_num_layers = config['predictor_fsmn_num_layers']
        num_memory_units = config['predictor_num_memory_units']
        ffn_inner_dim = config['predictor_ffn_inner_dim']
        dropout = config['predictor_dropout']
        shift = config['predictor_shift']
        lstm_units = config['predictor_lstm_units']

        dur_pred_prenet_units = config['dur_pred_prenet_units']
        dur_pred_lstm_units = config['dur_pred_lstm_units']

        self.pitch_predictor = VarFsmnRnnNARPredictor(
            input_dim,
            filter_size,
            fsmn_num_layers,
            num_memory_units,
            ffn_inner_dim,
            dropout,
            shift,
            lstm_units,
        )
        self.energy_predictor = VarFsmnRnnNARPredictor(
            input_dim,
            filter_size,
            fsmn_num_layers,
            num_memory_units,
            ffn_inner_dim,
            dropout,
            shift,
            lstm_units,
        )
        self.duration_predictor = VarRnnARPredictor(input_dim,
                                                    dur_pred_prenet_units,
                                                    dur_pred_lstm_units)

        self.length_regulator = LengthRegulator(config['outputs_per_step'])
        self.dur_position_encoder = DurSinusoidalPositionEncoder(
            config['encoder_projection_units'], config['outputs_per_step'])

        self.pitch_emb = nn.Conv1d(
            1, config['encoder_projection_units'], kernel_size=9, padding=4)
        self.energy_emb = nn.Conv1d(
            1, config['encoder_projection_units'], kernel_size=9, padding=4)

    def forward(
        self,
        inputs_text_embedding,
        inputs_emo_embedding,
        inputs_spk_embedding,
        masks=None,
        output_masks=None,
        duration_targets=None,
        pitch_targets=None,
        energy_targets=None,
    ):

        batch_size = inputs_text_embedding.size(0)

        variance_predictor_inputs = torch.cat([
            inputs_text_embedding, inputs_spk_embedding, inputs_emo_embedding
        ],
                                              dim=-1)  # noqa

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

        inputs_text_embedding_aug = (
            inputs_text_embedding + pitch_embeddings + energy_embeddings)
        duration_predictor_cond = torch.cat(
            [
                inputs_text_embedding_aug, inputs_spk_embedding,
                inputs_emo_embedding
            ],
            dim=-1,
        )
        if duration_targets is not None:
            duration_predictor_go_frame = torch.zeros(batch_size, 1).to(
                inputs_text_embedding.device)
            duration_predictor_input = torch.cat([
                duration_predictor_go_frame, duration_targets[:, :-1].float()
            ],
                                                 dim=-1)  # noqa
            duration_predictor_input = torch.log(duration_predictor_input + 1)
            log_duration_predictions, _ = self.duration_predictor(
                duration_predictor_input.unsqueeze(-1),
                duration_predictor_cond,
                masks=masks,
            )
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

        return (
            LR_text_outputs,
            LR_emo_outputs,
            LR_spk_outputs,
            LR_length_rounded,
            log_duration_predictions,
            pitch_predictions,
            energy_predictions,
        )


class MelPNCADecoder(nn.Module):

    def __init__(self, config):
        super(MelPNCADecoder, self).__init__()

        prenet_units = config['decoder_prenet_units']
        nb_layers = config['decoder_num_layers']
        nb_heads = config['decoder_num_heads']
        d_model = config['decoder_num_units']
        d_head = d_model // nb_heads
        d_inner = config['decoder_ffn_inner_dim']
        dropout = config['decoder_dropout']
        dropout_attn = config['decoder_attention_dropout']
        dropout_relu = config['decoder_relu_dropout']
        outputs_per_step = config['outputs_per_step']

        d_mem = (
            config['encoder_projection_units'] * outputs_per_step
            + config['emotion_units'] + config['speaker_units'])
        d_mel = config['num_mels']

        self.d_mel = d_mel
        self.r = outputs_per_step
        self.nb_layers = nb_layers

        self.mel_dec = HybridAttentionDecoder(
            d_mel,
            prenet_units,
            nb_layers,
            d_model,
            d_mem,
            nb_heads,
            d_head,
            d_inner,
            dropout,
            dropout_attn,
            dropout_relu,
            d_mel * outputs_per_step,
        )

    def forward(
        self,
        memory,
        x_band_width,
        h_band_width,
        target=None,
        mask=None,
        return_attns=False,
    ):
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
                return_attns=return_attns,
            )

        else:
            dec_output = []
            dec_pnca_attn_x_list = [[] for _ in range(self.nb_layers)]
            dec_pnca_attn_h_list = [[] for _ in range(self.nb_layers)]
            self.mel_dec.reset_state()
            input = go_frame
            for step in range(memory.size(1)):
                (
                    dec_output_step,
                    dec_pnca_attn_x_step,
                    dec_pnca_attn_h_step,
                ) = self.mel_dec.infer(
                    step,
                    input,
                    memory,
                    x_band_width,
                    h_band_width,
                    mask=mask,
                    return_attns=return_attns,
                )
                input = dec_output_step[:, :, -self.d_mel:]

                dec_output.append(dec_output_step)
                for layer_id, (pnca_x_attn, pnca_h_attn) in enumerate(
                        zip(dec_pnca_attn_x_step, dec_pnca_attn_h_step)):
                    left = memory.size(1) - pnca_x_attn.size(-1)
                    if left > 0:
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

        self.filter_size = config['postnet_filter_size']
        self.fsmn_num_layers = config['postnet_fsmn_num_layers']
        self.num_memory_units = config['postnet_num_memory_units']
        self.ffn_inner_dim = config['postnet_ffn_inner_dim']
        self.dropout = config['postnet_dropout']
        self.shift = config['postnet_shift']
        self.lstm_units = config['postnet_lstm_units']
        self.num_mels = config['num_mels']

        self.fsmn = FsmnEncoderV2(
            self.filter_size,
            self.fsmn_num_layers,
            self.num_mels,
            self.num_memory_units,
            self.ffn_inner_dim,
            self.dropout,
            self.shift,
        )
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


def average_frame_feat(pitch, durs):
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = F.pad(durs_cums_ends[:, :-1], (1, 0))
    pitch_nonzero_cums = F.pad(torch.cumsum(pitch != 0.0, dim=2), (1, 0))
    pitch_cums = F.pad(torch.cumsum(pitch, dim=2), (1, 0))

    bs, lengths = durs_cums_ends.size()
    n_formants = pitch.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, lengths)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, lengths)

    pitch_sums = (torch.gather(pitch_cums, 2, dce)
                  - torch.gather(pitch_cums, 2, dcs)).float()
    pitch_nelems = (torch.gather(pitch_nonzero_cums, 2, dce)
                    - torch.gather(pitch_nonzero_cums, 2, dcs)).float()

    pitch_avg = torch.where(pitch_nelems == 0.0, pitch_nelems,
                            pitch_sums / pitch_nelems)
    return pitch_avg


class FP_Predictor(nn.Module):

    def __init__(self, config):
        super(FP_Predictor, self).__init__()

        self.w_1 = nn.Conv1d(
            config['encoder_projection_units'],
            config['embedding_dim'] // 2,
            kernel_size=3,
            padding=1,
        )
        self.w_2 = nn.Conv1d(
            config['embedding_dim'] // 2,
            config['encoder_projection_units'],
            kernel_size=1,
            padding=0,
        )
        self.layer_norm1 = nn.LayerNorm(config['embedding_dim'] // 2, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(
            config['encoder_projection_units'], eps=1e-6)
        self.dropout_inner = nn.Dropout(0.1)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(config['encoder_projection_units'], 4)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.w_1(x))
        x = x.transpose(1, 2)
        x = self.dropout_inner(self.layer_norm1(x))
        x = x.transpose(1, 2)
        x = F.relu(self.w_2(x))
        x = x.transpose(1, 2)
        x = self.dropout(self.layer_norm2(x))
        output = F.softmax(self.fc(x), dim=2)
        return output


class KanTtsSAMBERT(nn.Module):

    def __init__(self, config):
        super(KanTtsSAMBERT, self).__init__()

        self.text_encoder = TextFftEncoder(config)
        self.spk_tokenizer = nn.Embedding(config['speaker'],
                                          config['speaker_units'])
        self.emo_tokenizer = nn.Embedding(config['emotion'],
                                          config['emotion_units'])
        self.variance_adaptor = VarianceAdaptor(config)
        self.mel_decoder = MelPNCADecoder(config)
        self.mel_postnet = PostNet(config)
        self.MAS = False
        if config.get('MAS', False):
            self.MAS = True
            self.align_attention = ConvAttention(
                n_mel_channels=config['num_mels'],
                n_text_channels=config['embedding_dim'],
                n_att_channels=config['num_mels'],
            )
        self.fp_enable = config.get('FP', False)
        if self.fp_enable:
            self.FP_predictor = FP_Predictor(config)

    def get_lfr_mask_from_lengths(self, lengths, max_len):
        batch_size = lengths.size(0)
        # padding according to the outputs_per_step
        padded_lr_lengths = torch.zeros_like(lengths)
        for i in range(batch_size):
            len_item = int(lengths[i].item())
            padding = self.mel_decoder.r - len_item % self.mel_decoder.r
            if padding < self.mel_decoder.r:
                padded_lr_lengths[i] = (len_item
                                        + padding) // self.mel_decoder.r
            else:
                padded_lr_lengths[i] = len_item // self.mel_decoder.r

        return get_mask_from_lengths(
            padded_lr_lengths, max_len=max_len // self.mel_decoder.r)

    def binarize_attention_parallel(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS.
           These will no longer recieve a gradient.

        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = b_mas(
                attn_cpu,
                in_lens.cpu().numpy(),
                out_lens.cpu().numpy(),
                width=1)
        return torch.from_numpy(attn_out).to(attn.get_device())

    def insert_fp(
        self,
        text_hid,
        FP_p,
        fp_label,
        fp_dict,
        inputs_emotion,
        inputs_speaker,
        input_lengths,
        input_masks,
    ):

        en, _, _ = self.text_encoder(fp_dict[1], return_attns=True)
        a, _, _ = self.text_encoder(fp_dict[2], return_attns=True)
        e, _, _ = self.text_encoder(fp_dict[3], return_attns=True)

        en = en.squeeze()
        a = a.squeeze()
        e = e.squeeze()

        max_len_ori = max(input_lengths)
        if fp_label is None:
            input_masks_r = ~input_masks
            fp_mask = (FP_p == FP_p.max(dim=2,
                                        keepdim=True)[0]).to(dtype=torch.int32)
            fp_mask = fp_mask[:, :, 1:] * input_masks_r.unsqueeze(2).expand(
                -1, -1, 3)
            fp_number = torch.sum(torch.sum(fp_mask, dim=2), dim=1)
        else:
            fp_number = torch.sum((fp_label > 0), dim=1)
        inter_lengths = input_lengths + 3 * fp_number
        max_len = max(inter_lengths)

        delta = max_len - max_len_ori
        if delta > 0:
            if delta > text_hid.shape[1]:
                nrepeat = delta // text_hid.shape[1]
                bias = delta % text_hid.shape[1]
                text_hid = torch.cat((text_hid, text_hid.repeat(
                    1, nrepeat, 1), text_hid[:, :bias, :]), 1)
                inputs_emotion = torch.cat(
                    (
                        inputs_emotion,
                        inputs_emotion.repeat(1, nrepeat),
                        inputs_emotion[:, :bias],
                    ),
                    1,
                )
                inputs_speaker = torch.cat(
                    (
                        inputs_speaker,
                        inputs_speaker.repeat(1, nrepeat),
                        inputs_speaker[:, :bias],
                    ),
                    1,
                )
            else:
                text_hid = torch.cat((text_hid, text_hid[:, :delta, :]), 1)
                inputs_emotion = torch.cat(
                    (inputs_emotion, inputs_emotion[:, :delta]), 1)
                inputs_speaker = torch.cat(
                    (inputs_speaker, inputs_speaker[:, :delta]), 1)

        if fp_label is None:
            for i in range(fp_mask.shape[0]):
                for j in range(fp_mask.shape[1] - 1, -1, -1):
                    if fp_mask[i][j][0] == 1:
                        text_hid[i] = torch.cat(
                            (text_hid[i][:j], en, text_hid[i][j:-3]), 0)
                    elif fp_mask[i][j][1] == 1:
                        text_hid[i] = torch.cat(
                            (text_hid[i][:j], a, text_hid[i][j:-3]), 0)
                    elif fp_mask[i][j][2] == 1:
                        text_hid[i] = torch.cat(
                            (text_hid[i][:j], e, text_hid[i][j:-3]), 0)
        else:
            for i in range(fp_label.shape[0]):
                for j in range(fp_label.shape[1] - 1, -1, -1):
                    if fp_label[i][j] == 1:
                        text_hid[i] = torch.cat(
                            (text_hid[i][:j], en, text_hid[i][j:-3]), 0)
                    elif fp_label[i][j] == 2:
                        text_hid[i] = torch.cat(
                            (text_hid[i][:j], a, text_hid[i][j:-3]), 0)
                    elif fp_label[i][j] == 3:
                        text_hid[i] = torch.cat(
                            (text_hid[i][:j], e, text_hid[i][j:-3]), 0)
        return text_hid, inputs_emotion, inputs_speaker, inter_lengths

    def forward(
        self,
        inputs_ling,
        inputs_emotion,
        inputs_speaker,
        input_lengths,
        output_lengths=None,
        mel_targets=None,
        duration_targets=None,
        pitch_targets=None,
        energy_targets=None,
        attn_priors=None,
        fp_label=None,
    ):
        batch_size = inputs_ling.size(0)

        is_training = mel_targets is not None
        input_masks = get_mask_from_lengths(
            input_lengths, max_len=inputs_ling.size(1))

        text_hid, enc_sla_attn_lst, ling_embedding = self.text_encoder(
            inputs_ling, input_masks, return_attns=True)

        inter_lengths = input_lengths
        FP_p = None
        if self.fp_enable:
            FP_p = self.FP_predictor(text_hid)
            fp_dict = self.fp_dict
            text_hid, inputs_emotion, inputs_speaker, inter_lengths = self.insert_fp(
                text_hid,
                FP_p,
                fp_label,
                fp_dict,
                inputs_emotion,
                inputs_speaker,
                input_lengths,
                input_masks,
            )

        # Monotonic-Alignment-Search
        if self.MAS and is_training:
            attn_soft, attn_logprob = self.align_attention(
                mel_targets.permute(0, 2, 1),
                ling_embedding.permute(0, 2, 1),
                input_masks,
                attn_priors,
            )
            attn_hard = self.binarize_attention_parallel(
                attn_soft, input_lengths, output_lengths)
            attn_hard_dur = attn_hard.sum(2)[:, 0, :]
            duration_targets = attn_hard_dur
            assert torch.all(
                torch.eq(duration_targets.sum(dim=1), output_lengths))
            pitch_targets = average_frame_feat(
                pitch_targets.unsqueeze(1), duration_targets).squeeze(1)
            energy_targets = average_frame_feat(
                energy_targets.unsqueeze(1), duration_targets).squeeze(1)
            # Padding the POS length to make it sum equal to max rounded output length
            for i in range(batch_size):
                len_item = int(output_lengths[i].item())
                padding = mel_targets.size(1) - len_item
                duration_targets[i, input_lengths[i]] = padding

        emo_hid = self.emo_tokenizer(inputs_emotion)
        spk_hid = self.spk_tokenizer(inputs_speaker)

        inter_masks = get_mask_from_lengths(
            inter_lengths, max_len=text_hid.size(1))

        if output_lengths is not None:
            output_masks = get_mask_from_lengths(
                output_lengths, max_len=mel_targets.size(1))
        else:
            output_masks = None

        (
            LR_text_outputs,
            LR_emo_outputs,
            LR_spk_outputs,
            LR_length_rounded,
            log_duration_predictions,
            pitch_predictions,
            energy_predictions,
        ) = self.variance_adaptor(
            text_hid,
            emo_hid,
            spk_hid,
            masks=inter_masks,
            output_masks=output_masks,
            duration_targets=duration_targets,
            pitch_targets=pitch_targets,
            energy_targets=energy_targets,
        )

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
                duration_targets.float().masked_fill(inter_masks, 0).max()
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
            return_attns=True,
        )

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
            'energy_predictions': energy_predictions,
            'duration_targets': duration_targets,
            'pitch_targets': pitch_targets,
            'energy_targets': energy_targets,
            'fp_predictions': FP_p,
            'valid_inter_lengths': inter_lengths,
        }

        res['LR_text_outputs'] = LR_text_outputs
        res['LR_emo_outputs'] = LR_emo_outputs
        res['LR_spk_outputs'] = LR_spk_outputs

        if self.MAS and is_training:
            res['attn_soft'] = attn_soft
            res['attn_hard'] = attn_hard
            res['attn_logprob'] = attn_logprob

        return res


class KanTtsTextsyBERT(nn.Module):

    def __init__(self, config):
        super(KanTtsTextsyBERT, self).__init__()

        self.text_encoder = TextFftEncoder(config)
        delattr(self.text_encoder, 'ling_proj')
        self.fc = nn.Linear(self.text_encoder.d_model, config['sy'])

    def forward(self, inputs_ling, input_lengths):
        res = {}

        input_masks = get_mask_from_lengths(
            input_lengths, max_len=inputs_ling.size(1))

        text_hid, enc_sla_attn_lst = self.text_encoder(
            inputs_ling, input_masks, return_attns=True)
        logits = self.fc(text_hid)

        res['logits'] = logits
        res['enc_slf_attn_lst'] = enc_sla_attn_lst

        return res
