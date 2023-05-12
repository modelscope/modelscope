# Part of implementation is borrowed and modified from Mask2Former, publicly available at
# https://github.com/facebookresearch/Mask2Former.
import os
from typing import Any, Dict, List

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
from .backbones import build_resnet_backbone
from .fastinst.fastinst_decoder import FastInstDecoder
from .fastinst.fastinst_encoder import PyramidPoolingModuleFPN

logger = get_logger()


@MODELS.register_module(Tasks.image_segmentation, module_name=Models.fastinst)
class FastInst(TorchModel):

    def __init__(self,
                 model_dir,
                 backbone=None,
                 encoder=None,
                 decoder=None,
                 pretrained=None,
                 classes=None,
                 **kwargs):
        """
        Deep Learning Technique for Human Parsing: A Survey and Outlook. See https://arxiv.org/abs/2301.00394
        Args:
            backbone (dict): backbone config.
            encoder (dict): encoder config.
            decoder (dict): decoder config.
            pretrained (bool): whether to use pretrained model
            classes (list): class names
        """
        super(FastInst, self).__init__(model_dir, **kwargs)

        self.backbone = build_resnet_backbone(
            **backbone, input_shape={'channels': 3})
        in_features = encoder.pop('in_features')
        input_shape = {
            k: v
            for k, v in self.backbone.output_shape().items()
            if k in in_features
        }
        encoder = PyramidPoolingModuleFPN(input_shape=input_shape, **encoder)
        decoder = FastInstDecoder(in_channels=encoder.convs_dim, **decoder)
        self.sem_seg_head = FastInstHead(
            pixel_decoder=encoder, transformer_predictor=decoder)

        self.num_classes = decoder.num_classes
        self.num_queries = decoder.num_queries
        self.size_divisibility = 32
        self.register_buffer(
            'pixel_mean',
            torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False)
        self.register_buffer(
            'pixel_std',
            torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)
        self.classes = classes
        self.test_topk_per_image = 100

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

    def forward(self, batched_inputs: List[dict]) -> Dict[str, Any]:
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

                mask_pred_result = self.sem_seg_postprocess(
                    mask_pred_result, image_size, height, width)
                mask_cls_result = mask_cls_result.to(mask_pred_result)

                instance_r = self.instance_inference(mask_cls_result,
                                                     mask_pred_result)
                processed_results[-1]['instances'] = instance_r

        return dict(eval_result=processed_results)

    @property
    def device(self):
        return self.pixel_mean.device

    def sem_seg_postprocess(self, result, img_size, output_height,
                            output_width):
        result = result[:, :img_size[0], :img_size[1]].expand(1, -1, -1, -1)
        result = F.interpolate(
            result,
            size=(output_height, output_width),
            mode='bilinear',
            align_corners=False)[0]
        return result

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
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

        result = {'image_size': image_size}
        # mask (before sigmoid)
        mask_pred_sigmoid = mask_pred.sigmoid()
        result['pred_masks'] = (mask_pred_sigmoid > 0.5).float()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred_sigmoid.flatten(1)
                                 * result['pred_masks'].flatten(1)).sum(1) / (
                                     result['pred_masks'].flatten(1).sum(1)
                                     + 1e-6)
        result['scores'] = scores_per_image * mask_scores_per_image
        result['pred_classes'] = labels_per_image
        return result


class FastInstHead(nn.Module):

    def __init__(
            self,
            *,
            pixel_decoder: nn.Module,
            # extra parameters
            transformer_predictor: nn.Module):
        """
        NOTE: this interface is experimental.
        Args:
            pixel_decoder: the pixel decoder module
            transformer_predictor: the transformer decoder that makes prediction
        """
        super().__init__()
        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor

    def forward(self, features, targets=None):
        return self.layers(features, targets)

    def layers(self, features, targets=None):
        mask_features, multi_scale_features = self.pixel_decoder.forward_features(
            features)
        predictions = self.predictor(multi_scale_features, mask_features,
                                     targets)
        return predictions
