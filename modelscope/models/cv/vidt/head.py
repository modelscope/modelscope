# The implementation here is modified based on timm,
# originally Apache 2.0 License and publicly available at
# https://github.com/naver-ai/vidt/blob/vidt-plus/methods/vidt/detector.py

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Detector(nn.Module):
    """ This is a combination of "Swin with RAM" and a "Neck-free Deformable Decoder" """

    def __init__(
            self,
            backbone,
            transformer,
            num_classes,
            num_queries,
            aux_loss=False,
            with_box_refine=False,
            # The three additional techniques for ViDT+
            epff=None,  # (1) Efficient Pyramid Feature Fusion Module
            with_vector=False,
            processor_dct=None,
            vector_hidden_dim=256,  # (2) UQR Module
            iou_aware=False,
            token_label=False,  # (3) Additional losses
            distil=False):
        """ Initializes the model.
        Args:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries (i.e., det tokens). This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            epff: None or fusion module available
            iou_aware: True if iou_aware is to be used.
              see the original paper https://arxiv.org/abs/1912.05992
            token_label: True if token_label is to be used.
              see the original paper https://arxiv.org/abs/2104.10858
            distil: whether to use knowledge distillation with token matching
        """

        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # two essential techniques used [default use]
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine

        # For UQR module for ViDT+
        self.with_vector = with_vector
        self.processor_dct = processor_dct
        if self.with_vector:
            print(
                f'Training with vector_hidden_dim {vector_hidden_dim}.',
                flush=True)
            self.vector_embed = MLP(hidden_dim, vector_hidden_dim,
                                    self.processor_dct.n_keep, 3)

        # For two additional losses for ViDT+
        self.iou_aware = iou_aware
        self.token_label = token_label

        # distillation
        self.distil = distil

        # For EPFF module for ViDT+
        if epff is None:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        # This is 1x1 conv -> so linear layer
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
            self.input_proj = nn.ModuleList(input_proj_list)

            # initialize the projection layer for [PATCH] tokens
            for proj in self.input_proj:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)
            self.fusion = None
        else:
            # the cross scale fusion module has its own reduction layers
            self.fusion = epff

        # channel dim reduction for [DET] tokens
        self.tgt_proj = nn.Sequential(
            # This is 1x1 conv -> so linear layer
            nn.Conv2d(backbone.num_channels[-2], hidden_dim, kernel_size=1),
            nn.GroupNorm(32, hidden_dim),
        )

        # channel dim reductionfor [DET] learnable pos encodings
        self.query_pos_proj = nn.Sequential(
            # This is 1x1 conv -> so linear layer
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(32, hidden_dim),
        )

        # initialize detection head: box regression and classification
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # initialize projection layer for [DET] tokens and encodings
        nn.init.xavier_uniform_(self.tgt_proj[0].weight, gain=1)
        nn.init.constant_(self.tgt_proj[0].bias, 0)
        nn.init.xavier_uniform_(self.query_pos_proj[0].weight, gain=1)
        nn.init.constant_(self.query_pos_proj[0].bias, 0)

        if self.with_vector:
            nn.init.constant_(self.vector_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.vector_embed.layers[-1].bias.data, 0)

        # the prediction is made for each decoding layers + the standalone detector (Swin with RAM)
        num_pred = transformer.decoder.num_layers + 1

        # set up all required nn.Module for additional techniques
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:],
                              -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList(
                [self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList(
                [self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        if self.with_vector:
            nn.init.constant_(self.vector_embed.layers[-1].bias.data[2:], -2.0)
            self.vector_embed = nn.ModuleList(
                [self.vector_embed for _ in range(num_pred)])

        if self.iou_aware:
            self.iou_embed = MLP(hidden_dim, hidden_dim, 1, 3)
            if with_box_refine:
                self.iou_embed = _get_clones(self.iou_embed, num_pred)
            else:
                self.iou_embed = nn.ModuleList(
                    [self.iou_embed for _ in range(num_pred)])

    def forward(self, features_0, features_1, features_2, features_3, det_tgt,
                det_pos, mask):
        """ The forward step of ViDT

        Args:
            The forward expects a NestedTensor, which consists of:
            - features_0: images feature
            - features_1: images feature
            - features_2: images feature
            - features_3: images feature
            - det_tgt: images det logits feature
            - det_pos: images det position feature
            - mask: images mask
        Returns:
            A dictionary having the key and value pairs below:
            - "out_pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
            - "out_pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
        """
        features = [features_0, features_1, features_2, features_3]

        # [DET] token and encoding projection to compact representation for the input to the Neck-free transformer
        det_tgt = self.tgt_proj(det_tgt.unsqueeze(-1)).squeeze(-1).permute(
            0, 2, 1)
        det_pos = self.query_pos_proj(
            det_pos.unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)

        # [PATCH] token projection
        shapes = []
        for le, src in enumerate(features):
            shapes.append(src.shape[-2:])

        srcs = []
        if self.fusion is None:
            for le, src in enumerate(features):
                srcs.append(self.input_proj[le](src))
        else:
            # EPFF (multi-scale fusion) is used if fusion is activated
            srcs = self.fusion(features)

        masks = []
        for le, src in enumerate(srcs):
            # resize mask
            shapes.append(src.shape[-2:])
            _mask = F.interpolate(
                mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
            masks.append(_mask)
            assert mask is not None

        outputs_classes = []
        outputs_coords = []

        # return the output of the neck-free decoder
        hs, init_reference, inter_references, enc_token_class_unflat = self.transformer(
            srcs, masks, det_tgt, det_pos)

        # perform predictions via the detection head
        for lvl in range(hs.shape[0]):
            reference = init_reference if lvl == 0 else inter_references[lvl
                                                                         - 1]
            reference = inverse_sigmoid(reference)

            outputs_class = self.class_embed[lvl](hs[lvl])
            # bbox output + reference
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        # stack all predictions made from each decoding layers
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        outputs_vector = None
        if self.with_vector:
            outputs_vectors = []
            for lvl in range(hs.shape[0]):
                outputs_vector = self.vector_embed[lvl](hs[lvl])
                outputs_vectors.append(outputs_vector)
            outputs_vector = torch.stack(outputs_vectors)

        # final prediction is made the last decoding layer
        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1]
        }

        if self.with_vector:
            out.update({'pred_vectors': outputs_vector[-1]})

        # aux loss is defined by using the rest predictions
        if self.aux_loss and self.transformer.decoder.num_layers > 0:
            out['aux_outputs'] = self._set_aux_loss(outputs_class,
                                                    outputs_coord,
                                                    outputs_vector)

        # iou awareness loss is defined for each decoding layer similar to auxiliary decoding loss
        if self.iou_aware:
            outputs_ious = []
            for lvl in range(hs.shape[0]):
                outputs_ious.append(self.iou_embed[lvl](hs[lvl]))
            outputs_iou = torch.stack(outputs_ious)
            out['pred_ious'] = outputs_iou[-1]

            if self.aux_loss:
                for i, aux in enumerate(out['aux_outputs']):
                    aux['pred_ious'] = outputs_iou[i]

        # token label loss
        if self.token_label:
            out['enc_tokens'] = {'pred_logits': enc_token_class_unflat}

        if self.distil:
            # 'patch_token': multi-scale patch tokens from each stage
            # 'body_det_token' and 'neck_det_tgt': the input det_token for multiple detection heads
            out['distil_tokens'] = {
                'patch_token': srcs,
                'body_det_token': det_tgt,
                'neck_det_token': hs
            }

        out_pred_logits = out['pred_logits']
        out_pred_boxes = out['pred_boxes']
        return out_pred_logits, out_pred_boxes

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_vector):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.

        if outputs_vector is None:
            return [{
                'pred_logits': a,
                'pred_boxes': b
            } for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        else:
            return [{
                'pred_logits': a,
                'pred_boxes': b,
                'pred_vectors': c
            } for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1],
                                 outputs_vector[:-1])]


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


# process post_results
def get_predictions(post_results, bbox_thu=0.40):
    batch_final_res = []
    for per_img_res in post_results:
        per_img_final_res = []
        for i in range(len(per_img_res['scores'])):
            score = float(per_img_res['scores'][i].cpu())
            label = int(per_img_res['labels'][i].cpu())
            bbox = []
            for it in per_img_res['boxes'][i].cpu():
                bbox.append(int(it))
            if score >= bbox_thu:
                per_img_final_res.append([score, label, bbox])
        batch_final_res.append(per_img_final_res)
    return batch_final_res


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, processor_dct=None):
        super().__init__()
        # For instance segmentation using UQR module
        self.processor_dct = processor_dct

    @torch.no_grad()
    def forward(self, out_logits, out_bbox, target_sizes):
        """ Perform the computation

        Args:
            out_logits: raw logits outputs of the model
            out_bbox: raw bbox outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1,
                             topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h],
                                dim=1).to(torch.float32)
        boxes = boxes * scale_fct[:, None, :]

        results = [{
            'scores': s,
            'labels': l,
            'boxes': b
        } for s, l, b in zip(scores, labels, boxes)]

        return results


def _get_clones(module, N):
    """ Clone a moudle N times """

    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
