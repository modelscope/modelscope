# Copyright (c) OpenMMLab. All rights reserved.
# Implementation in this file is modified based on mmdetection
# Originally Apache 2.0 License and publicly available at https://github.com/open-mmlab/mmdetection
import torch
from mmcv.runner import force_fp32
from mmdet.models.builder import ROI_EXTRACTORS
from mmdet.models.roi_heads.roi_extractors.base_roi_extractor import \
    BaseRoIExtractor


@ROI_EXTRACTORS.register_module()
class SingleRoINExtractor(BaseRoIExtractor):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (List[int]): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56,
                 init_cfg=None,
                 gc_context=False,
                 offset_feature=False):
        super(SingleRoINExtractor, self).__init__(roi_layer, out_channels,
                                                  featmap_strides, init_cfg)
        self.finest_scale = finest_scale
        self.gc_context = gc_context
        self.offset_feature = offset_feature
        self.pool = torch.nn.AdaptiveAvgPool2d(7)

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        a = rois[:, 3] - rois[:, 1]
        b = rois[:, 4] - rois[:, 2]
        scale = torch.sqrt(a * b)
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function."""
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        expand_dims = (-1, self.out_channels * out_size[0] * out_size[1])
        if torch.onnx.is_in_onnx_export():
            # Work around to export mask-rcnn to onnx
            roi_feats = rois[:, :1].clone().detach()
            roi_feats = roi_feats.expand(*expand_dims)
            roi_feats = roi_feats.reshape(-1, self.out_channels, *out_size)
            roi_feats = roi_feats * 0
        else:
            roi_feats = feats[0].new_zeros(
                rois.size(0), self.out_channels, *out_size)
        # TODO: remove this when parrots supports
        if torch.__version__ == 'parrots':
            roi_feats.requires_grad = True

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        if self.gc_context:
            context = []
            for feat in feats:
                context.append(self.pool(feat))

        batch_size = feats[0].shape[0]
        target_lvls = self.map_roi_levels(rois, num_levels)

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            mask = target_lvls == i
            if torch.onnx.is_in_onnx_export():
                # To keep all roi_align nodes exported to onnx
                # and skip nonzero op
                mask = mask.float().unsqueeze(-1)
                # select target level rois and reset the rest rois to zero.
                rois_i = rois.clone().detach()
                rois_i *= mask
                mask_exp = mask.expand(*expand_dims).reshape(roi_feats.shape)
                roi_feats_t = self.roi_layers[i](feats[i], rois_i)
                roi_feats_t *= mask_exp
                roi_feats += roi_feats_t
                continue
            inds = mask.nonzero(as_tuple=False).squeeze(1)
            if inds.numel() > 0:
                rois_ = rois[inds]
                # todo offset
                rois_offset = rois[inds]
                offset = torch.zeros(rois_.size(0), 5)
                _, _, x_max, y_max = rois_[:, 1].min().item(), rois_[:, 2].min(
                ).item(), rois_[:, 3].max().item(), rois_[:, 4].max().item()
                offset[:, 1:3] = -100 * torch.ones(rois_.size(0), 1)
                offset[:, 3:5] = 100 * torch.ones(rois_.size(0), 1)
                rois_offset += offset.cuda()
                rois_offset_thsxy = torch.clamp(rois_offset[:, 1:3], min=0.)
                rois_offset_ths_xmax = torch.clamp(
                    rois_offset[:, 3], max=x_max)
                rois_offset_ths_ymax = torch.clamp(
                    rois_offset[:, 4], max=y_max)
                rois_offset[:, 1:3] = rois_offset_thsxy
                rois_offset[:,
                            3], rois_offset[:,
                                            4] = rois_offset_ths_xmax, rois_offset_ths_ymax
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats_t_offset = self.roi_layers[i](feats[i], rois_offset)
                if self.gc_context:
                    for j in range(batch_size):
                        roi_feats_t[rois_[:, 0] == j] += context[i][j]
                elif self.offset_feature:
                    roi_feats_t += roi_feats_t_offset

                roi_feats[inds] = roi_feats_t
            else:
                # Sometimes some pyramid levels will not be used for RoI
                # feature extraction and this will cause an incomplete
                # computation graph in one GPU, which is different from those
                # in other GPUs and will cause a hanging error.
                # Therefore, we add it to ensure each feature pyramid is
                # included in the computation graph to avoid runtime bugs.
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.
        return roi_feats
