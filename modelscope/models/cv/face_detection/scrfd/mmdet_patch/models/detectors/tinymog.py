"""
The implementation here is modified based on insightface, originally MIT license and publicly avaialbe at
https://github.com/deepinsight/insightface/tree/master/detection/scrfd/mmdet/models/detectors/scrfd.py
"""
import torch
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.single_stage import SingleStageDetector

from ....mmdet_patch.core.bbox import bbox2result


@DETECTORS.register_module()
class TinyMog(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TinyMog, self).__init__(backbone, neck, bbox_head, train_cfg,
                                      test_cfg, pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_keypointss=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_keypointss,
                                              gt_bboxes_ignore)
        return losses

    def simple_test(self,
                    img,
                    img_metas,
                    rescale=False,
                    repeat_head=1,
                    output_kps_var=0,
                    output_results=1):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
            repeat_head (int): repeat inference times in head
            output_kps_var (int): whether output kps var to calculate quality
            output_results (int): 0: nothing  1: bbox  2: both bbox and kps

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        assert repeat_head >= 1
        kps_out0 = []
        kps_out1 = []
        kps_out2 = []
        for i in range(repeat_head):
            outs = self.bbox_head(x)
            kps_out0 += [outs[2][0].detach().cpu().numpy()]
            kps_out1 += [outs[2][1].detach().cpu().numpy()]
            kps_out2 += [outs[2][2].detach().cpu().numpy()]
        if output_kps_var:
            var0 = np.var(np.vstack(kps_out0), axis=0).mean()
            var1 = np.var(np.vstack(kps_out1), axis=0).mean()
            var2 = np.var(np.vstack(kps_out2), axis=0).mean()
            var = np.mean([var0, var1, var2])
        else:
            var = None

        if output_results > 0:
            if torch.onnx.is_in_onnx_export():
                cls_score, bbox_pred, kps_pred = outs
                for c in cls_score:
                    print(c.shape)
                for c in bbox_pred:
                    print(c.shape)
                if self.bbox_head.use_kps:
                    for c in kps_pred:
                        print(c.shape)
                    return (cls_score, bbox_pred, kps_pred)
                else:
                    return (cls_score, bbox_pred)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_metas, rescale=rescale)

            # return kps if use_kps
            if len(bbox_list[0]) == 2:
                bbox_results = [
                    bbox2result(det_bboxes, det_labels,
                                self.bbox_head.num_classes)
                    for det_bboxes, det_labels in bbox_list
                ]
            elif len(bbox_list[0]) == 3:
                if output_results == 2:
                    bbox_results = [
                        bbox2result(
                            det_bboxes,
                            det_labels,
                            self.bbox_head.num_classes,
                            kps=det_kps,
                            num_kps=self.bbox_head.NK)
                        for det_bboxes, det_labels, det_kps in bbox_list
                    ]
                elif output_results == 1:
                    bbox_results = [
                        bbox2result(det_bboxes, det_labels,
                                    self.bbox_head.num_classes)
                        for det_bboxes, det_labels, _ in bbox_list
                    ]
        else:
            bbox_results = None
        if var is not None:
            return bbox_results, var
        else:
            return bbox_results

    def feature_test(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs
