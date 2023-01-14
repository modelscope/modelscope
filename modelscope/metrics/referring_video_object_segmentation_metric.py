# Part of the implementation is borrowed and modified from MTTR,
# publicly available at https://github.com/mttr2021/MTTR
from typing import Dict

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import decode
from tqdm import tqdm

from modelscope.metainfo import Metrics
from modelscope.utils.registry import default_group
from .base import Metric
from .builder import METRICS, MetricKeys


@METRICS.register_module(
    group_key=default_group,
    module_name=Metrics.referring_video_object_segmentation_metric)
class ReferringVideoObjectSegmentationMetric(Metric):
    """The metric computation class for movie scene segmentation classes.
    """

    def __init__(self,
                 ann_file=None,
                 calculate_precision_and_iou_metrics=True):
        self.ann_file = ann_file
        self.calculate_precision_and_iou_metrics = calculate_precision_and_iou_metrics
        self.preds = []

    def add(self, outputs: Dict, inputs: Dict):
        preds_batch = outputs['pred']
        self.preds.extend(preds_batch)

    def evaluate(self):
        coco_gt = COCO(self.ann_file)
        coco_pred = coco_gt.loadRes(self.preds)
        coco_eval = COCOeval(coco_gt, coco_pred, iouType='segm')
        coco_eval.params.useCats = 0

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        ap_labels = [
            'mAP 0.5:0.95', 'AP 0.5', 'AP 0.75', 'AP 0.5:0.95 S',
            'AP 0.5:0.95 M', 'AP 0.5:0.95 L'
        ]
        ap_metrics = coco_eval.stats[:6]
        eval_metrics = {la: m for la, m in zip(ap_labels, ap_metrics)}
        if self.calculate_precision_and_iou_metrics:
            precision_at_k, overall_iou, mean_iou = calculate_precision_at_k_and_iou_metrics(
                coco_gt, coco_pred)
            eval_metrics.update({
                f'P@{k}': m
                for k, m in zip([0.5, 0.6, 0.7, 0.8, 0.9], precision_at_k)
            })
            eval_metrics.update({
                'overall_iou': overall_iou,
                'mean_iou': mean_iou
            })

        return eval_metrics

    def merge(self, other: 'ReferringVideoObjectSegmentationMetric'):
        self.preds.extend(other.preds)

    def __getstate__(self):
        return self.ann_file, self.calculate_precision_and_iou_metrics, self.preds

    def __setstate__(self, state):
        self.ann_file, self.calculate_precision_and_iou_metrics, self.preds = state


def compute_iou(outputs: torch.Tensor, labels: torch.Tensor, EPS=1e-6):
    outputs = outputs.int()
    intersection = (outputs & labels).float().sum(
        (1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum(
        (1, 2))  # Will be zero if both are 0
    iou = (intersection + EPS) / (union + EPS
                                  )  # EPS is used to avoid division by zero
    return iou, intersection, union


def calculate_precision_at_k_and_iou_metrics(coco_gt: COCO, coco_pred: COCO):
    print('evaluating precision@k & iou metrics...')
    counters_by_iou = {iou: 0 for iou in [0.5, 0.6, 0.7, 0.8, 0.9]}
    total_intersection_area = 0
    total_union_area = 0
    ious_list = []
    for instance in tqdm(coco_gt.imgs.keys()
                         ):  # each image_id contains exactly one instance
        gt_annot = coco_gt.imgToAnns[instance][0]
        gt_mask = decode(gt_annot['segmentation'])
        pred_annots = coco_pred.imgToAnns[instance]
        pred_annot = sorted(
            pred_annots,
            key=lambda a: a['score'])[-1]  # choose pred with highest score
        pred_mask = decode(pred_annot['segmentation'])
        iou, intersection, union = compute_iou(
            torch.tensor(pred_mask).unsqueeze(0),
            torch.tensor(gt_mask).unsqueeze(0))
        iou, intersection, union = iou.item(), intersection.item(), union.item(
        )
        for iou_threshold in counters_by_iou.keys():
            if iou > iou_threshold:
                counters_by_iou[iou_threshold] += 1
        total_intersection_area += intersection
        total_union_area += union
        ious_list.append(iou)
    num_samples = len(ious_list)
    precision_at_k = np.array(list(counters_by_iou.values())) / num_samples
    overall_iou = total_intersection_area / total_union_area
    mean_iou = np.mean(ious_list)
    return precision_at_k, overall_iou, mean_iou
