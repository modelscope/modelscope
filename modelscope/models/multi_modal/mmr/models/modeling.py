# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

import os
import platform
from collections import OrderedDict
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from modelscope.models.multi_modal.mmr.models.module_clip import (
    _PT_NAME, CLIP, QuickGELU, convert_weights)
from modelscope.models.multi_modal.mmr.models.module_cross import \
    Transformer as TransformerClip
from modelscope.models.multi_modal.mmr.models.until_module import (AllGather,
                                                                   CrossEn,
                                                                   LayerNorm)
from modelscope.utils.logger import get_logger

allgather = AllGather.apply

logger = get_logger()
__all__ = ['CLIP4Clip']


class CLIP4Clip(nn.Module):

    def __init__(self, config):
        super(CLIP4Clip, self).__init__()

        self.config = config
        self.loose_type = config['loose_type']
        self.sim_header = config['sim_header']
        if self.sim_header in [
                'tightTransf', 'tightFc1', 'tightFc2', 'tightFc3', 'tightFc4',
                'tightMean', 'tightFc5'
        ]:
            assert self.loose_type is False

        backbone = config['pretrained_clip_name']

        # fix backbone without downlond
        model_path = '{}/ViT-B-16.pt'.format(config['model_dir'])
        if not os.path.exists(model_path):
            logger.info('no model loaded!!!')

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location='cpu').eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location='cpu')

        vision_width = state_dict['visual.conv1.weight'].shape[0]
        vision_layers = len([
            k for k in state_dict.keys()
            if k.startswith('visual.') and k.endswith('.attn.in_proj_weight')
        ])
        vision_patch_size = state_dict['visual.conv1.weight'].shape[-1]
        grid_size = round(
            (state_dict['visual.positional_embedding'].shape[0] - 1)**0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = state_dict['text_projection'].shape[1]
        context_length = state_dict['positional_embedding'].shape[0]
        vocab_size = state_dict['token_embedding.weight'].shape[0]
        transformer_width = state_dict['ln_final.weight'].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(
            set(
                k.split('.')[2] for k in state_dict
                if k.startswith('transformer.resblocks')))

        cut_top_layer = 0
        self.clip = CLIP(
            embed_dim,
            image_resolution,
            vision_layers - cut_top_layer,
            vision_width,
            vision_patch_size,
            context_length,
            vocab_size,
            transformer_width,
            transformer_heads,
            transformer_layers - cut_top_layer,
            linear_patch=config['linear_patch'],
            use_gc=config['use_gc']).float()

        if backbone in ['ViT-B/32', 'ViT-B/16']:
            cross_config = SimpleNamespace(**{
                'hidden_size': 512,
                'max_position_embeddings': 128,
            })
        elif backbone in ['ViT-L/14', 'ViT-B/14-336px']:
            cross_config = SimpleNamespace(**{
                'hidden_size': 768,
                'max_position_embeddings': 128,
            })
        else:
            raise ValueError

        cross_config.max_position_embeddings = context_length
        self.cross_config = cross_config

        self.text_weight_fc = nn.Sequential(
            nn.Linear(transformer_width, transformer_width),
            nn.ReLU(inplace=True), nn.Linear(transformer_width, 1))
        self.video_weight_fc = nn.Sequential(
            nn.Linear(transformer_width, transformer_width),
            nn.ReLU(inplace=True), nn.Linear(transformer_width, 1))

        if self.loose_type is False:
            raise NotImplementedError

        if self.sim_header in ['seqLSTM', 'seqTransf', 'tightFc1']:
            self.frame_position_embeddings = nn.Embedding(
                cross_config.max_position_embeddings, cross_config.hidden_size)
        if self.sim_header in ['seqTransf', 'tightFc1']:
            self.transformerClip = TransformerClip(
                width=transformer_width,
                layers=config['cross_num_hidden_layers'],
                heads=transformer_heads,
            )
        if self.sim_header == 'seqLSTM':
            self.lstm_visual = nn.LSTM(
                input_size=cross_config.hidden_size,
                hidden_size=cross_config.hidden_size,
                batch_first=True,
                bidirectional=False,
                num_layers=1)

        self.loss_fct = CrossEn(config)

        self.apply(self.init_weights)
        self.clip.load_state_dict(state_dict, strict=False)

        # ===> Initialization trick [HARD CODE]
        if backbone not in _PT_NAME:
            raise NotImplementedError
            # reload
        else:
            if config['linear_patch'] == '3d':
                raise NotImplementedError

        new_state_dict = OrderedDict()
        if self.sim_header == 'tightTransf':
            raise NotImplementedError

        if self.sim_header in ['seqLSTM', 'seqTransf', 'seqFc1']:
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find('frame_position_embeddings') > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in state_dict.items():
                    if key == 'positional_embedding':
                        new_state_dict[
                            'frame_position_embeddings.weight'] = val.clone()
                        continue
                    if self.sim_header in [
                            'seqTransf', 'seqFc1'
                    ] and key.find('transformer.resblocks') == 0:
                        num_layer = int(key.split('.')[2])
                        # cut from beginning
                        if num_layer < config['cross_num_hidden_layers']:
                            new_state_dict[key.replace(
                                'transformer.',
                                'transformerClip.')] = val.clone()
                            continue
        # <=== End of initialization trick

        self.load_state_dict(
            new_state_dict, strict=False
        )  # only update new state (seqTransf/seqLSTM/tightTransf)
        if self.sim_header == 'tightFc5':
            raise ValueError

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                video,
                video_mask=None):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        # B x T x 3 x H x W - >  (B x T) x 3 x H x W
        video = torch.as_tensor(video).float()
        if len(video.shape) == 6:  # image
            b, bs, ts, channel, h, w = video.shape
            b = b * bs
        else:  # video
            b, ts, channel, h, w = video.shape
        video = video.view(b * ts, channel, h, w)

        sequence_output, visual_output = self.get_sequence_visual_output(
            input_ids,
            token_type_ids,
            attention_mask,
            video,
            video_mask,
            shaped=True)

        if self.training:
            loss = 0.
            sim_matrix1, sim_matrix2, barlow_loss = self.get_similarity_logits(
                sequence_output,
                visual_output,
                attention_mask,
                video_mask,
                shaped=True,
                loose_type=self.loose_type)
            sim_loss = (self.loss_fct(sim_matrix1)
                        + self.loss_fct(sim_matrix2)) / 2
            loss += sim_loss + barlow_loss * self.config.cdcr_lambda

            return loss
        else:
            return None

    def get_sequence_output(self,
                            input_ids,
                            token_type_ids,
                            attention_mask,
                            shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_hidden = self.clip.encode_text(
            input_ids, return_hidden=True, prompt=None)[1].float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1,
                                               sequence_hidden.size(-1))

        return sequence_hidden

    def get_visual_output(self, video, video_mask, shaped=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, ts, channel, h, w = video.shape
            video = video.view(b * ts, channel, h, w)

        bs_pair = video_mask.size(0)
        visual_hidden = self.clip.encode_image(video).float()
        visual_hidden = visual_hidden.float().view(bs_pair, -1,
                                                   visual_hidden.size(-1))

        return visual_hidden

    def get_sequence_visual_output(self,
                                   input_ids,
                                   token_type_ids,
                                   attention_mask,
                                   video,
                                   video_mask,
                                   shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            if len(video.shape) == 6:  # image
                b, bs, ts, channel, h, w = video.shape
                b = b * bs
            else:  # video
                b, ts, channel, h, w = video.shape
            video = video.view(b * ts, channel, h, w)

        sequence_output = self.get_sequence_output(
            input_ids, token_type_ids, attention_mask, shaped=True)
        visual_output = self.get_visual_output(video, video_mask, shaped=True)

        return sequence_output, visual_output

    def agg_video_feat(self, visual_output, video_mask, sim_header='meanP'):
        if self.config.max_sum == 0:
            raise ValueError

        if sim_header == 'meanP':
            # Default: Parameter-free type
            pass
        elif sim_header == 'seqLSTM':
            # Sequential type: LSTM
            visual_output_original = visual_output
            visual_output = pack_padded_sequence(
                visual_output,
                torch.sum(video_mask, dim=-1).cpu(),
                batch_first=True,
                enforce_sorted=False)
            visual_output, _ = self.lstm_visual(visual_output)
            if self.training:
                self.lstm_visual.flatten_parameters()
            visual_output, _ = pad_packed_sequence(
                visual_output, batch_first=True)
            visual_output = torch.cat(
                (visual_output, visual_output_original[:,
                                                       visual_output.size(1):,
                                                       ...].contiguous()),
                dim=1)
            visual_output = visual_output + visual_output_original
        elif sim_header == 'seqTransf':
            # Sequential type: Transformer Encoder
            visual_output_original = visual_output
            seq_length = visual_output.size(1)
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=visual_output.device)
            position_ids = position_ids.unsqueeze(0).expand(
                visual_output.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(
                position_ids)
            visual_output = visual_output + frame_position_embeddings

            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(
                -1, video_mask.size(1), -1)
            visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
            visual_output = self.transformerClip(visual_output,
                                                 extended_video_mask)
            visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
            visual_output = visual_output + visual_output_original

        return visual_output

    def wti_interaction(self, text_feat, video_feat, text_mask, video_mask):
        text_weight = self.text_weight_fc(text_feat).squeeze(
            2)  # B x N_t x D -> B x N_t
        text_weight.masked_fill_(
            torch.tensor((1 - text_mask), dtype=torch.bool), float('-inf'))
        text_weight = torch.softmax(text_weight, dim=-1)  # B x N_t

        video_weight = self.video_weight_fc(video_feat).squeeze(
            2)  # B x N_v x D -> B x N_v
        video_weight.masked_fill_(
            torch.tensor((1 - video_mask), dtype=torch.bool), float('-inf'))
        video_weight = torch.softmax(video_weight, dim=-1)  # B x N_v

        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)

        retrieve_logits = torch.einsum('atd,bvd->abtv',
                                       [text_feat, video_feat])
        retrieve_logits = torch.einsum('abtv,at->abtv',
                                       [retrieve_logits, text_mask])
        retrieve_logits = torch.einsum('abtv,bv->abtv',
                                       [retrieve_logits, video_mask])

        t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
        t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_weight])

        v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
        v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, video_weight])
        retrieve_logits = (t2v_logits + v2t_logits) / 2.0

        if self.training:
            logit_scale = self.clip.logit_scale.exp()
            retrieve_logits = logit_scale * retrieve_logits

            # selecet max
            max_idx1 = max_idx1[torch.arange(max_idx1.shape[0]),
                                torch.arange(max_idx1.shape[1])]
            max_idx2 = max_idx2[torch.arange(max_idx2.shape[0]),
                                torch.arange(max_idx2.shape[1])]

            max_t_feat = text_feat[torch.arange(max_idx2.shape[0]).
                                   repeat_interleave(max_idx2.shape[1]),
                                   max_idx2.flatten()].squeeze(1)
            max_v_feat = video_feat[torch.arange(max_idx1.shape[0]).
                                    repeat_interleave(max_idx1.shape[1]),
                                    max_idx1.flatten()].squeeze(1)

            t_feat = text_feat.reshape(-1, text_feat.shape[-1])
            t_mask = text_mask.flatten().type(torch.bool)
            v_feat = video_feat.reshape(-1, video_feat.shape[-1])
            v_mask = video_mask.flatten().type(torch.bool)
            t_feat = t_feat[t_mask]
            v_feat = v_feat[v_mask]
            max_t_feat = max_t_feat[v_mask]
            max_v_feat = max_v_feat[t_mask]
            text_weight = text_weight.flatten()[t_mask]
            video_weight = video_weight.flatten()[v_mask]

            z_a_norm = (t_feat - t_feat.mean(0)) / t_feat.std(0)  # (BxN_t)xD
            z_b_norm = (max_v_feat - max_v_feat.mean(0)) / max_v_feat.std(
                0)  # (BxN_t)xD

            x_a_norm = (v_feat - v_feat.mean(0)) / v_feat.std(0)  # (BxN_v)xD
            x_b_norm = (max_t_feat - max_t_feat.mean(0)) / max_t_feat.std(
                0)  # (BxN_v)xD

            # cross-correlation matrix
            N, D = z_a_norm.shape
            B = text_feat.shape[0]
            c1 = torch.einsum('acd,a->cd',
                              torch.einsum('ac,ad->acd', z_a_norm, z_b_norm),
                              text_weight) / B  # DxD
            c2 = torch.einsum('acd,a->cd',
                              torch.einsum('ac,ad->acd', x_a_norm, x_b_norm),
                              video_weight) / B  # DxD
            c = (c1 + c2) / 2.0
            # loss
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = c.flatten()[1:].view(D - 1, D + 1)[:, :-1].pow_(2).sum()
            cdcr_loss = (
                on_diag * self.config.cdcr_alpha1
                + off_diag * self.config.cdcr_alpha2)
            return retrieve_logits, retrieve_logits.T, cdcr_loss
        else:
            return retrieve_logits, retrieve_logits.T

    def _loose_similarity(self,
                          sequence_output,
                          visual_output,
                          attention_mask,
                          video_mask,
                          sim_header='seqTransf'):
        sequence_output, visual_output = sequence_output.contiguous(
        ), visual_output.contiguous()

        visual_output = self.agg_video_feat(visual_output, video_mask,
                                            sim_header)

        if self.training:  # batch merge here
            visual_output = allgather(visual_output, self.config)
            attention_mask = allgather(attention_mask, self.config)
            video_mask = allgather(video_mask, self.config)
            sequence_output = allgather(sequence_output, self.config)
            torch.distributed.barrier()  # force sync

        return self.wti_interaction(sequence_output, visual_output,
                                    attention_mask, video_mask)

    def get_similarity_logits(self,
                              sequence_output,
                              visual_output,
                              attention_mask,
                              video_mask,
                              shaped=False,
                              loose_type=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        if loose_type:
            assert self.sim_header in ['meanP', 'seqLSTM', 'seqTransf']

            if self.training:
                retrieve_logits1, retrieve_logits2, barlow_loss = self._loose_similarity(
                    sequence_output,
                    visual_output,
                    attention_mask,
                    video_mask,
                    sim_header=self.sim_header)
                return retrieve_logits1, retrieve_logits2, barlow_loss
            else:
                retrieve_logits1, retrieve_logits2 = self._loose_similarity(
                    sequence_output,
                    visual_output,
                    attention_mask,
                    video_mask,
                    sim_header=self.sim_header)
                return retrieve_logits1, retrieve_logits2
        else:
            raise NotImplementedError

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items()
                          if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
