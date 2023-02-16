# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict, Mapping, Union

from modelscope.metainfo import Metrics
from modelscope.utils.config import ConfigDict
from modelscope.utils.constant import Tasks
from modelscope.utils.registry import Registry, build_from_cfg, default_group

METRICS = Registry('metrics')


class MetricKeys(object):
    ACCURACY = 'accuracy'
    F1 = 'f1'
    Binary_F1 = 'binary-f1'
    Macro_F1 = 'macro-f1'
    Micro_F1 = 'micro-f1'
    PRECISION = 'precision'
    RECALL = 'recall'
    PSNR = 'psnr'
    SSIM = 'ssim'
    LPIPS = 'lpips'
    NIQE = 'niqe'
    AVERAGE_LOSS = 'avg_loss'
    FScore = 'fscore'
    FID = 'fid'
    BLEU_1 = 'bleu-1'
    BLEU_4 = 'bleu-4'
    ROUGE_1 = 'rouge-1'
    ROUGE_L = 'rouge-l'
    NED = 'ned'  # ocr metric
    mAP = 'mAP'
    BatchAcc = 'inbatch_t2i_recall_at_1'
    CROPPING_RATIO = 'cropping_ratio'
    DISTORTION_VALUE = 'distortion_value'
    STABILITY_SCORE = 'stability_score'
    PPL = 'ppl'
    PLCC = 'plcc'
    SRCC = 'srcc'
    RMSE = 'rmse'
    MRR = 'mrr'
    NDCG = 'ndcg'


task_default_metrics = {
    Tasks.image_segmentation: [Metrics.image_ins_seg_coco_metric],
    Tasks.sentence_similarity: [Metrics.seq_cls_metric],
    Tasks.nli: [Metrics.seq_cls_metric],
    Tasks.sentiment_classification: [Metrics.seq_cls_metric],
    Tasks.token_classification: [Metrics.token_cls_metric],
    Tasks.text_generation: [Metrics.text_gen_metric],
    Tasks.text_classification: [Metrics.seq_cls_metric],
    Tasks.image_denoising: [Metrics.image_denoise_metric],
    Tasks.image_deblurring: [Metrics.image_denoise_metric],
    Tasks.video_super_resolution: [Metrics.video_super_resolution_metric],
    Tasks.image_color_enhancement: [Metrics.image_color_enhance_metric],
    Tasks.image_portrait_enhancement:
    [Metrics.image_portrait_enhancement_metric],
    Tasks.video_summarization: [Metrics.video_summarization_metric],
    Tasks.image_captioning: [Metrics.accuracy],
    Tasks.visual_question_answering: [Metrics.accuracy],
    Tasks.movie_scene_segmentation: [Metrics.movie_scene_segmentation_metric],
    Tasks.image_inpainting: [Metrics.image_inpainting_metric],
    Tasks.referring_video_object_segmentation:
    [Metrics.referring_video_object_segmentation_metric],
    Tasks.video_frame_interpolation:
    [Metrics.video_frame_interpolation_metric],
    Tasks.video_stabilization: [Metrics.video_stabilization_metric],
    Tasks.image_quality_assessment_degradation:
    [Metrics.image_quality_assessment_degradation_metric],
    Tasks.image_quality_assessment_mos:
    [Metrics.image_quality_assessment_mos_metric],
}


def build_metric(metric_cfg: Union[str, Dict],
                 field: str = default_group,
                 default_args: dict = None):
    """ Build metric given metric_name and field.

    Args:
        metric_name (str | dict): The metric name or metric config dict.
        field (str, optional):  The field of this metric, default value: 'default' for all fields.
        default_args (dict, optional): Default initialization arguments.
    """
    if isinstance(metric_cfg, Mapping):
        assert 'type' in metric_cfg
    else:
        metric_cfg = ConfigDict({'type': metric_cfg})
    return build_from_cfg(
        metric_cfg, METRICS, group_key=field, default_args=default_args)
