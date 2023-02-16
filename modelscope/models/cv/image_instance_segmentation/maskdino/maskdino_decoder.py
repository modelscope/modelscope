# The implementation is adopted from Mask DINO, made publicly available under the Apache License,
# Version 2.0 at https://github.com/IDEA-Research/MaskDINO
# Part of implementation is borrowed from Mask2Former,
# https://github.com/facebookresearch/Mask2Former, under MIT license.

import torch
from torch import nn

from .dino_decoder import DeformableTransformerDecoderLayer, TransformerDecoder
from .utils import (MLP, Conv2d, box_xyxy_to_cxcywh,
                    gen_encoder_output_proposals, get_bounding_boxes,
                    inverse_sigmoid)


class MaskDINODecoder(nn.Module):

    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        mask_dim: int,
        enforce_input_project: bool,
        two_stage: bool,
        initialize_box_type: bool,
        initial_pred: bool,
        learn_tgt: bool,
        total_num_feature_levels: int = 4,
        dropout: float = 0.0,
        activation: str = 'relu',
        nhead: int = 8,
        dec_n_points: int = 4,
        return_intermediate_dec: bool = True,
        query_dim: int = 4,
        dec_layer_share: bool = False,
        semantic_ce_loss: bool = False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            dec_layers: number of Transformer decoder layers
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
            dropout: dropout rate
            activation: activation function
            nhead: num heads in multi-head attention
            dec_n_points: number of sampling points in decoder
            return_intermediate_dec: return the intermediate results of decoder
            query_dim: 4 -> (x, y, w, h)
            dec_layer_share: whether to share each decoder layer
            semantic_ce_loss: use ce loss for semantic segmentation
        """
        super().__init__()

        assert mask_classification, 'Only support mask classification model'
        self.mask_classification = mask_classification
        self.num_feature_levels = total_num_feature_levels
        self.initial_pred = initial_pred

        # define Transformer decoder here
        self.learn_tgt = learn_tgt
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.two_stage = two_stage
        self.initialize_box_type = initialize_box_type
        self.total_num_feature_levels = total_num_feature_levels

        self.num_queries = num_queries
        self.semantic_ce_loss = semantic_ce_loss
        # learnable query features
        if not two_stage or self.learn_tgt:
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
        if not two_stage and initialize_box_type == 'no':
            self.query_embed = nn.Embedding(num_queries, 4)
        if two_stage:
            self.enc_output = nn.Linear(hidden_dim, hidden_dim)
            self.enc_output_norm = nn.LayerNorm(hidden_dim)

        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(
                    Conv2d(in_channels, hidden_dim, kernel_size=1))
                nn.init.kaiming_uniform_(self.input_proj[-1].weight, a=1)
                nn.init.constant_(self.input_proj[-1].bias, 0)
            else:
                self.input_proj.append(nn.Sequential())
        self.num_classes = num_classes
        # output FFNs
        assert self.mask_classification, 'why not class embedding?'
        if self.mask_classification:
            if self.semantic_ce_loss:
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            else:
                self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.label_enc = nn.Embedding(num_classes, hidden_dim)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # init decoder
        self.decoder_norm = decoder_norm = nn.LayerNorm(hidden_dim)
        decoder_layer = DeformableTransformerDecoderLayer(
            hidden_dim, dim_feedforward, dropout, activation,
            self.num_feature_levels, nhead, dec_n_points)
        self.decoder = TransformerDecoder(
            decoder_layer,
            self.num_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=hidden_dim,
            query_dim=query_dim,
            num_feature_levels=self.num_feature_levels,
            dec_layer_share=dec_layer_share,
        )

        self.hidden_dim = hidden_dim
        self._bbox_embed = _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        box_embed_layerlist = [_bbox_embed for i in range(self.num_layers)
                               ]  # share box prediction each layer
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.decoder.bbox_embed = self.bbox_embed

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def pred_box(self, reference, hs, ref0=None):
        """
        Args:
            reference: reference box coordinates from each decoder layer
            hs: content
            ref0: whether there are prediction from the first layer
        """
        if ref0 is None:
            outputs_coord_list = []
        else:
            outputs_coord_list = [ref0]
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
                zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(
                layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)
        return outputs_coord_list

    def forward(self, x, mask_features, masks, targets=None):
        """
        Args:
            x: input, a list of multi-scale feature
            mask_features: is the per-pixel embeddings with resolution 1/4 of the original image,
                obtained by fusing backbone encoder encoded features. This is used to produce binary masks.
            masks: mask in the original image
            targets: used for denoising training
        """
        assert len(x) == self.num_feature_levels
        size_list = []
        # disable mask, it does not affect performance
        enable_mask = 0
        if masks is not None:
            for src in x:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [
                torch.zeros((src.size(0), src.size(2), src.size(3)),
                            device=src.device,
                            dtype=torch.bool) for src in x
            ]
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for i in range(self.num_feature_levels):
            idx = self.num_feature_levels - 1 - i
            bs, c, h, w = x[idx].shape
            size_list.append(x[i].shape[-2:])
            spatial_shapes.append(x[idx].shape[-2:])
            src_flatten.append(self.input_proj[idx](
                x[idx]).flatten(2).transpose(1, 2))
            mask_flatten.append(masks[i].flatten(1))
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        predictions_class = []
        predictions_mask = []
        if self.two_stage:
            output_memory, output_proposals = gen_encoder_output_proposals(
                src_flatten, mask_flatten, spatial_shapes)
            output_memory = self.enc_output_norm(
                self.enc_output(output_memory))
            enc_outputs_class_unselected = self.class_embed(output_memory)
            enc_outputs_coord_unselected = self._bbox_embed(
                output_memory
            ) + output_proposals  # (bs, \sum{hw}, 4) unsigmoid
            topk = self.num_queries
            topk_proposals = torch.topk(
                enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]
            refpoint_embed_undetach = torch.gather(
                enc_outputs_coord_unselected, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4))  # unsigmoid
            refpoint_embed = refpoint_embed_undetach.detach()

            tgt_undetach = torch.gather(output_memory, 1,
                                        topk_proposals.unsqueeze(-1).repeat(
                                            1, 1,
                                            self.hidden_dim))  # unsigmoid

            outputs_class, outputs_mask = self.forward_prediction_heads(
                tgt_undetach.transpose(0, 1), mask_features)
            tgt = tgt_undetach.detach()
            if self.learn_tgt:
                tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            interm_outputs = dict()
            interm_outputs['pred_logits'] = outputs_class
            interm_outputs['pred_boxes'] = refpoint_embed_undetach.sigmoid()
            interm_outputs['pred_masks'] = outputs_mask

            if self.initialize_box_type != 'no':
                # convert masks into boxes to better initialize box in the decoder
                assert self.initial_pred
                flatten_mask = outputs_mask.detach().flatten(0, 1)
                h, w = outputs_mask.shape[-2:]
                if self.initialize_box_type == 'bitmask':  # slower, but more accurate
                    refpoint_embed = get_bounding_boxes(flatten_mask > 0)
                else:
                    assert NotImplementedError
                refpoint_embed = box_xyxy_to_cxcywh(
                    refpoint_embed) / torch.as_tensor(
                        [w, h, w, h],
                        dtype=torch.float,
                        device=refpoint_embed.device)
                refpoint_embed = refpoint_embed.reshape(
                    outputs_mask.shape[0], outputs_mask.shape[1], 4)
                refpoint_embed = inverse_sigmoid(refpoint_embed)
        elif not self.two_stage:
            tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            refpoint_embed = self.query_embed.weight[None].repeat(bs, 1, 1)

        tgt_mask = None
        mask_dict = None

        # direct prediction from the matching and denoising part in the begining
        if self.initial_pred:
            outputs_class, outputs_mask = self.forward_prediction_heads(
                tgt.transpose(0, 1), mask_features, self.training)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=src_flatten.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=None,
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=tgt_mask)
        for i, output in enumerate(hs):
            outputs_class, outputs_mask = self.forward_prediction_heads(
                output.transpose(0, 1), mask_features, self.training
                or (i == len(hs) - 1))
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        # iteratively box prediction
        if self.initial_pred:
            out_boxes = self.pred_box(references, hs, refpoint_embed.sigmoid())
            assert len(predictions_class) == self.num_layers + 1
        else:
            out_boxes = self.pred_box(references, hs)
        if mask_dict is not None:
            predictions_mask = torch.stack(predictions_mask)
            predictions_class = torch.stack(predictions_class)
            predictions_class, out_boxes, predictions_mask = \
                self.dn_post_process(predictions_class, out_boxes, mask_dict, predictions_mask)
            predictions_class, predictions_mask = list(
                predictions_class), list(predictions_mask)
        elif self.training:  # this is to insure self.label_enc participate in the model
            predictions_class[-1] += 0.0 * self.label_enc.weight.sum()

        out = {
            'pred_logits':
            predictions_class[-1],
            'pred_masks':
            predictions_mask[-1],
            'pred_boxes':
            out_boxes[-1],
            'aux_outputs':
            self._set_aux_loss(
                predictions_class if self.mask_classification else None,
                predictions_mask, out_boxes)
        }
        if self.two_stage:
            out['interm_outputs'] = interm_outputs
        return out, mask_dict

    def forward_prediction_heads(self, output, mask_features, pred_mask=True):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        outputs_mask = None
        if pred_mask:
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum('bqc,bchw->bqhw', mask_embed,
                                        mask_features)

        return outputs_class, outputs_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, out_boxes=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if out_boxes is None:
            return [{
                'pred_logits': a,
                'pred_masks': b
            } for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]
        else:
            return [{
                'pred_logits': a,
                'pred_masks': b,
                'pred_boxes': c
            } for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1],
                                 out_boxes[:-1])]
