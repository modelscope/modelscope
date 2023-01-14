# Part of the implementation is borrowed and modified from Mask DINO, publicly available at
# https://github.com/IDEA-Research/MaskDINO
# Part of implementation is borrowed and modified from Mask2Former, publicly available at
# https://github.com/facebookresearch/Mask2Former.

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.models.cv.image_instance_segmentation.backbones import \
    D2SwinTransformer
from modelscope.utils.constant import ModelFile
from modelscope.utils.logger import get_logger
from .maskdino.maskdino_decoder import MaskDINODecoder
from .maskdino.maskdino_encoder import MaskDINOEncoder

logger = get_logger()


class MaskDINOSwin(nn.Module):

    def __init__(self, backbone, encoder, decoder, pretrained=None, **kwargs):
        """
        Mask DINO: Towards A Unified Transformer-based Framework for Object
            Detection and Segmentation. See https://arxiv.org/abs/2206.02777
        Args:
            backbone (dict): backbone config.
            encoder (dict): encoder config.
            decoder (dict): decoder config.
            pretrained (bool): whether to use pretrained model
        """
        super(MaskDINOSwin, self).__init__()
        self.register_buffer(
            'pixel_mean',
            torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False)
        self.register_buffer(
            'pixel_std',
            torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)
        self.size_divisibility = 32

        self.backbone = D2SwinTransformer(**backbone)
        input_shape = {
            k: v
            for k, v in self.backbone.output_shape().items()
            if k in encoder['transformer_in_features']
        }
        encoder = MaskDINOEncoder(input_shape=input_shape, **encoder)
        decoder = MaskDINODecoder(**decoder)
        self.sem_seg_head = MaskDINOHead(
            pixel_decoder=encoder, transformer_predictor=decoder)
        self.num_classes = decoder.num_classes
        self.num_queries = decoder.num_queries
        self.test_topk_per_image = 100

        self.classes = kwargs.pop('classes', None)

        if pretrained:
            assert 'model_dir' in kwargs, 'pretrained model dir is missing.'
            model_path = os.path.join(kwargs['model_dir'],
                                      ModelFile.TORCH_MODEL_FILE)
            logger.info(f'loading model from {model_path}')
            weight = torch.load(model_path)['model']
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
                        logger.info(f'size mismatch for {name}, skip loading.')
                        del weight[name]
                else:
                    logger.info(
                        f'{name} doesn\'t exist in current model, skip loading.'
                    )

            self.load_state_dict(weight, strict=False)
            logger.info('load model done')

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, **kwargs):

        images = [x['image'].to(self.device) for x in batched_inputs]
        images = [(255. * x - self.pixel_mean) / self.pixel_std
                  for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)

        if self.training:
            raise NotImplementedError
        else:
            outputs, _ = self.sem_seg_head(features)
            mask_cls_results = outputs['pred_logits']
            mask_pred_results = outputs['pred_masks']
            mask_box_results = outputs['pred_boxes']
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode='bilinear',
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, mask_box_result, input_per_image, image_size in zip(
                    mask_cls_results, mask_pred_results, mask_box_results,
                    batched_inputs, images.image_sizes
            ):  # image_size is augmented size, not divisible to 32
                height = input_per_image.get('height',
                                             image_size[0])  # real size
                width = input_per_image.get('width', image_size[1])
                processed_results.append({})
                new_size = mask_pred_result.shape[
                    -2:]  # padded size (divisible to 32)

                # post process
                mask_pred_result = mask_pred_result[:, :image_size[0], :
                                                    image_size[1]].expand(
                                                        1, -1, -1, -1)
                mask_pred_result = F.interpolate(
                    mask_pred_result,
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False)[0]

                mask_cls_result = mask_cls_result.to(mask_pred_result)
                mask_box_result = mask_box_result.to(mask_pred_result)
                height = new_size[0] / image_size[0] * height
                width = new_size[1] / image_size[1] * width
                mask_box_result = self.box_postprocess(mask_box_result, height,
                                                       width)

                instance_r = self.instance_inference(mask_cls_result,
                                                     mask_pred_result,
                                                     mask_box_result)
                processed_results[-1]['instances'] = instance_r

            return dict(eval_result=processed_results)

    def instance_inference(self, mask_cls, mask_pred, mask_box_result):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        scores = mask_cls.sigmoid()  # [100, 80]
        labels = torch.arange(
            self.num_classes,
            device=self.device).unsqueeze(0).repeat(self.num_queries,
                                                    1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(
            self.test_topk_per_image, sorted=False)  # select 100
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // self.num_classes
        mask_pred = mask_pred[topk_indices]
        # if this is panoptic segmentation, we only keep the "thing" classes

        result = {'image_size': image_size}
        # mask (before sigmoid)
        result['pred_masks'] = (mask_pred > 0).float()
        # half mask box half pred box
        mask_box_result = mask_box_result[topk_indices]
        result['pred_boxes'] = mask_box_result

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1)
                                 * result['pred_masks'].flatten(1)).sum(1) / (
                                     result['pred_masks'].flatten(1).sum(1)
                                     + 1e-6)
        result['scores'] = scores_per_image * mask_scores_per_image
        result['pred_classes'] = labels_per_image
        return result

    def box_postprocess(self, out_bbox, img_h, img_w):
        # postprocess box height and width
        x_c, y_c, w, h = out_bbox.unbind(-1)
        boxes = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w),
                 (y_c + 0.5 * h)]
        boxes = torch.stack(boxes, dim=-1)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
        scale_fct = scale_fct.to(out_bbox)
        boxes = boxes * scale_fct
        return boxes


class MaskDINOHead(nn.Module):

    def __init__(
        self,
        pixel_decoder: nn.Module,
        transformer_predictor: nn.Module,
    ):
        super().__init__()
        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor

    def forward(self, features, mask=None, targets=None):
        return self.layers(features, mask, targets=targets)

    def layers(self, features, mask=None, targets=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            features, mask)
        predictions = self.predictor(
            multi_scale_features, mask_features, mask, targets=targets)
        return predictions


class ImageList(object):

    def __init__(self, tensor, image_sizes):
        self.tensor = tensor
        self.image_sizes = image_sizes

    def __len__(self):
        return len(self.image_sizes)

    def __getitem__(self, idx):
        size = self.image_sizes[idx]
        return self.tensor[idx, ..., :size[0], :size[1]]

    @torch.jit.unused
    def to(self, *args, **kwargs):
        cast_tensor = self.tensor.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)

    @property
    def device(self):
        return self.tensor.device

    @staticmethod
    def from_tensors(tensors, size_divisibility=0, pad_value=0.0):
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))
        for t in tensors:
            assert isinstance(t, torch.Tensor), type(t)
            assert t.shape[:-2] == tensors[0].shape[:-2], t.shape

        image_sizes = [(im.shape[-2], im.shape[-1]) for im in tensors]
        image_sizes_tensor = [torch.as_tensor(x) for x in image_sizes]
        max_size = torch.stack(image_sizes_tensor).max(0).values

        if size_divisibility > 1:
            stride = size_divisibility
            # the last two dims are H,W, both subject to divisibility requirement
            max_size = (max_size + (stride - 1)) // stride * stride

        # handle weirdness of scripting and tracing ...
        if torch.jit.is_scripting():
            max_size = max_size.to(dtype=torch.long).tolist()
        else:
            if torch.jit.is_tracing():
                image_sizes = image_sizes_tensor

        if len(tensors) == 1:
            image_size = image_sizes[0]
            padding_size = [
                0, max_size[-1] - image_size[1], 0,
                max_size[-2] - image_size[0]
            ]
            batched_imgs = F.pad(
                tensors[0], padding_size, value=pad_value).unsqueeze_(0)
        else:
            # max_size can be a tensor in tracing mode, therefore convert to list
            batch_shape = [len(tensors)] + list(
                tensors[0].shape[:-2]) + list(max_size)
            batched_imgs = tensors[0].new_full(batch_shape, pad_value)
            for img, pad_img in zip(tensors, batched_imgs):
                pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)

        return ImageList(batched_imgs.contiguous(), image_sizes)
