# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.metainfo import Metrics
from modelscope.utils.config import ConfigDict
from modelscope.utils.constant import Tasks
from modelscope.utils.registry import Registry, build_from_cfg, default_group

METRICS = Registry('metrics')


class MetricKeys(object):
    ACCURACY = 'accuracy'
    F1 = 'f1'
    PRECISION = 'precision'
    RECALL = 'recall'
    PSNR = 'psnr'
    SSIM = 'ssim'


task_default_metrics = {
    Tasks.image_segmentation: [Metrics.image_ins_seg_coco_metric],
    Tasks.sentence_similarity: [Metrics.seq_cls_metric],
    Tasks.nli: [Metrics.seq_cls_metric],
    Tasks.sentiment_classification: [Metrics.seq_cls_metric],
    Tasks.token_classification: [Metrics.token_cls_metric],
    Tasks.text_generation: [Metrics.text_gen_metric],
    Tasks.image_denoising: [Metrics.image_denoise_metric],
    Tasks.image_color_enhancement: [Metrics.image_color_enhance_metric],
    Tasks.image_portrait_enhancement:
    [Metrics.image_portrait_enhancement_metric],
}


def build_metric(metric_name: str,
                 field: str = default_group,
                 default_args: dict = None):
    """ Build metric given metric_name and field.

    Args:
        metric_name (:obj:`str`): The metric name.
        field (str, optional):  The field of this metric, default value: 'default' for all fields.
        default_args (dict, optional): Default initialization arguments.
    """
    cfg = ConfigDict({'type': metric_name})
    return build_from_cfg(
        cfg, METRICS, group_key=field, default_args=default_args)
