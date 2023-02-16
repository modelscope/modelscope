# Part of the implementation is borrowed and modified from M2FP, made publicly available
# under the CC BY-NC 4.0 License at https://github.com/soeaver/M2FP
import os
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.image_instance_segmentation.maskdino_swin import \
    ImageList
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .backbone import build_resnet_deeplab_backbone
from .m2fp.m2fp_decoder import MultiScaleMaskedTransformerDecoder
from .m2fp.m2fp_encoder import MSDeformAttnPixelDecoder

logger = get_logger()


@MODELS.register_module(Tasks.image_segmentation, module_name=Models.m2fp)
class M2FP(TorchModel):

    def __init__(self,
                 model_dir,
                 backbone=None,
                 encoder=None,
                 decoder=None,
                 pretrained=None,
                 input_single_human=None,
                 classes=None,
                 num_parsing=None,
                 single_human=True,
                 parsing_ins_score_thr=0.5,
                 parsing_on=False,
                 semantic_on=True,
                 sem_seg_postprocess_before_inference=True,
                 **kwargs):
        """
        Deep Learning Technique for Human Parsing: A Survey and Outlook. See https://arxiv.org/abs/2301.00394
        Args:
            backbone (dict): backbone config.
            encoder (dict): encoder config.
            decoder (dict): decoder config.
            pretrained (bool): whether to use pretrained model
            input_single_human (dict): input size config for single human parsing
            classes (list): class names
            num_parsing (int): total number of parsing instances, only for multiple human parsing
            single_human (bool): whether the task is single human parsing
            parsing_ins_score_thr: instance score threshold for multiple human parsing
            parsing_on (bool): whether to parse results, only for multiple human parsing
            semantic_on (bool): whether to output semantic map
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
        """
        super(M2FP, self).__init__(model_dir, **kwargs)

        self.register_buffer(
            'pixel_mean',
            torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False)
        self.register_buffer(
            'pixel_std',
            torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)
        self.size_divisibility = 32

        self.backbone = build_resnet_deeplab_backbone(
            **backbone, input_shape={'channels': 3})
        in_features = encoder.pop('in_features')
        input_shape = {
            k: v
            for k, v in self.backbone.output_shape().items()
            if k in in_features
        }
        encoder = MSDeformAttnPixelDecoder(input_shape=input_shape, **encoder)
        decoder = MultiScaleMaskedTransformerDecoder(
            in_channels=encoder.conv_dim, **decoder)
        self.sem_seg_head = M2FPHead(
            pixel_decoder=encoder, transformer_predictor=decoder)
        self.num_classes = decoder.num_classes
        self.num_queries = decoder.num_queries
        self.test_topk_per_image = 100

        self.input_single_human = input_single_human
        self.classes = classes
        self.num_parsing = num_parsing
        self.single_human = single_human
        self.parsing_ins_score_thr = parsing_ins_score_thr
        self.parsing_on = parsing_on
        self.semantic_on = semantic_on
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference or parsing_on

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        if pretrained:
            model_path = os.path.join(model_dir, ModelFile.TORCH_MODEL_FILE)
            logger.info(f'loading model from {model_path}')
            weight = torch.load(model_path, map_location='cpu')['model']
            tgt_weight = self.state_dict()
            for name in list(weight.keys()):
                if name in tgt_weight:
                    load_size = weight[name].size()
                    tgt_size = tgt_weight[name].size()
                    mis_match = False
                    if len(load_size) != len(tgt_size):
                        mis_match = True
                    else:
                        for n1, n2 in zip(load_size, tgt_size):
                            if n1 != n2:
                                mis_match = True
                                break
                    if mis_match:
                        logger.info(
                            f'size mismatch for {name} '
                            f'({load_size} -> {tgt_size}), skip loading.')
                        del weight[name]
                else:
                    logger.info(
                        f'{name} doesn\'t exist in current model, skip loading.'
                    )

            self.load_state_dict(weight, strict=False)
            logger.info('load model done')

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        batched_inputs = input['batched_inputs']
        images = [x['image'].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        return dict(
            outputs=outputs, batched_inputs=batched_inputs, images=images)

    def postprocess(self, input: Dict[str, Any]) -> Dict[str, Any]:
        outputs = input['outputs']
        batched_inputs = input['batched_inputs']
        images = input['images']
        if self.training:
            raise NotImplementedError
        else:
            mask_cls_results = outputs['pred_logits']  # (B, Q, C+1)
            mask_pred_results = outputs['pred_masks']  # (B, Q, H, W)
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode='bilinear',
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                    mask_cls_results, mask_pred_results, batched_inputs,
                    images.image_sizes):
                height = input_per_image.get('height', image_size[0])
                width = input_per_image.get('width', image_size[1])
                processed_results.append({})  # for each image

                if self.sem_seg_postprocess_before_inference:
                    if not self.single_human:
                        mask_pred_result = self.sem_seg_postprocess(
                            mask_pred_result, image_size, height, width)
                    else:
                        mask_pred_result = self.single_human_sem_seg_postprocess(
                            mask_pred_result, image_size,
                            input_per_image['crop_box'], height, width)
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = self.semantic_inference(mask_cls_result,
                                                mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        if not self.single_human:
                            r = self.sem_seg_postprocess(
                                r, image_size, height, width)
                        else:
                            r = self.single_human_sem_seg_postprocess(
                                r, image_size, input_per_image['crop_box'],
                                height, width)
                        processed_results[-1]['sem_seg'] = r

                # parsing inference
                if self.parsing_on:
                    parsing_r = self.instance_parsing_inference(
                        mask_cls_result, mask_pred_result)
                    processed_results[-1]['parsing'] = parsing_r

        return dict(eval_result=processed_results)

    @property
    def device(self):
        return self.pixel_mean.device

    def single_human_sem_seg_postprocess(self, result, img_size, crop_box,
                                         output_height, output_width):
        result = result[:, :img_size[0], :img_size[1]]
        result = result[:, crop_box[1]:crop_box[3],
                        crop_box[0]:crop_box[2]].expand(1, -1, -1, -1)
        result = F.interpolate(
            result,
            size=(output_height, output_width),
            mode='bilinear',
            align_corners=False)[0]
        return result

    def sem_seg_postprocess(self, result, img_size, output_height,
                            output_width):
        result = result[:, :img_size[0], :img_size[1]].expand(1, -1, -1, -1)
        result = F.interpolate(
            result,
            size=(output_height, output_width),
            mode='bilinear',
            align_corners=False)[0]
        return result

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(
            mask_cls, dim=-1)[..., :-1]  # discard non-sense category
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum('qc,qhw->chw', mask_cls, mask_pred)
        return semseg

    def instance_parsing_inference(self, mask_cls, mask_pred):
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(
            self.num_classes,
            device=self.device).unsqueeze(0).repeat(self.num_queries,
                                                    1).flatten(0, 1)

        scores_per_image, topk_indices = scores.flatten(0, 1).topk(
            self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.num_classes
        mask_pred = mask_pred[topk_indices]

        binary_pred_masks = (mask_pred > 0).float()
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * binary_pred_masks.flatten(1)).sum(1) / \
                                (binary_pred_masks.flatten(1).sum(1) + 1e-6)

        pred_scores = scores_per_image * mask_scores_per_image
        pred_labels = labels_per_image
        pred_masks = mask_pred

        # prepare outputs
        part_instance_res = []
        human_instance_res = []

        # bkg and part instances
        bkg_part_index = torch.where(pred_labels != self.num_parsing)[0]
        bkg_part_labels = pred_labels[bkg_part_index]
        bkg_part_scores = pred_scores[bkg_part_index]
        bkg_part_masks = pred_masks[bkg_part_index, :, :]

        # human instances
        human_index = torch.where(pred_labels == self.num_parsing)[0]
        human_labels = pred_labels[human_index]
        human_scores = pred_scores[human_index]
        human_masks = pred_masks[human_index, :, :]

        semantic_res = self.paste_instance_to_semseg_probs(
            bkg_part_labels, bkg_part_scores, bkg_part_masks)

        # part instances
        part_index = torch.where(bkg_part_labels != 0)[0]
        part_labels = bkg_part_labels[part_index]
        part_scores = bkg_part_scores[part_index]
        part_masks = bkg_part_masks[part_index, :, :]

        # part instance results
        for idx in range(part_labels.shape[0]):
            if part_scores[idx] < 0.1:
                continue
            part_instance_res.append({
                'category_id':
                part_labels[idx].cpu().tolist(),
                'score':
                part_scores[idx].cpu().tolist(),
                'mask':
                part_masks[idx],
            })

        # human instance results
        for human_idx in range(human_scores.shape[0]):
            if human_scores[human_idx] > 0.1:
                human_instance_res.append({
                    'category_id':
                    human_labels[human_idx].cpu().tolist(),
                    'score':
                    human_scores[human_idx].cpu().tolist(),
                    'mask':
                    human_masks[human_idx],
                })

        return {
            'semantic_outputs': semantic_res,
            'part_outputs': part_instance_res,
            'human_outputs': human_instance_res,
        }

    def paste_instance_to_semseg_probs(self, labels, scores, mask_probs):
        im_h, im_w = mask_probs.shape[-2:]
        semseg_im = []
        for cls_ind in range(self.num_parsing):
            cate_inds = torch.where(labels == cls_ind)[0]
            cate_scores = scores[cate_inds]
            cate_mask_probs = mask_probs[cate_inds, :, :].sigmoid()
            semseg_im.append(
                self.paste_category_probs(cate_scores, cate_mask_probs, im_h,
                                          im_w))

        return torch.stack(semseg_im, dim=0)

    def paste_category_probs(self, scores, mask_probs, h, w):
        category_probs = torch.zeros((h, w),
                                     dtype=torch.float32,
                                     device=mask_probs.device)
        paste_times = torch.zeros((h, w),
                                  dtype=torch.float32,
                                  device=mask_probs.device)

        index = scores.argsort()
        for k in range(len(index)):
            if scores[index[k]] < self.parsing_ins_score_thr:
                continue
            ins_mask_probs = mask_probs[index[k], :, :] * scores[index[k]]
            category_probs = torch.where(ins_mask_probs > 0.5,
                                         ins_mask_probs + category_probs,
                                         category_probs)
            paste_times += torch.where(ins_mask_probs > 0.5, 1, 0)

        paste_times = torch.where(paste_times == 0, paste_times + 1,
                                  paste_times)
        category_probs /= paste_times

        return category_probs


class M2FPHead(nn.Module):

    def __init__(self, pixel_decoder: nn.Module,
                 transformer_predictor: nn.Module):
        super().__init__()
        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor

    def forward(self, features, mask=None):
        return self.layers(features, mask)

    def layers(self, features, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            features)
        predictions = self.predictor(multi_scale_features, mask_features, mask)
        return predictions
