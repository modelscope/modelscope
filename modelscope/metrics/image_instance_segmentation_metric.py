# Part of the implementation is borrowed and modified from MMDetection, publicly available at
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/coco.py
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Any, Dict

import numpy as np
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from modelscope.fileio import dump, load
from modelscope.metainfo import Metrics
from modelscope.metrics import METRICS, Metric
from modelscope.utils.registry import default_group


@METRICS.register_module(
    group_key=default_group, module_name=Metrics.image_ins_seg_coco_metric)
class ImageInstanceSegmentationCOCOMetric(Metric):
    """The metric computation class for COCO-style image instance segmentation.
    """

    def __init__(self):
        self.ann_file = None
        self.classes = None
        self.metrics = ['bbox', 'segm']
        self.proposal_nums = (100, 300, 1000)
        self.iou_thrs = np.linspace(
            .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.results = []

    def add(self, outputs: Dict[str, Any], inputs: Dict[str, Any]):
        result = outputs['eval_result']
        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        self.results.extend(result)
        if self.ann_file is None:
            self.ann_file = outputs['img_metas'][0]['ann_file']
            self.classes = outputs['img_metas'][0]['classes']

    def evaluate(self):
        cocoGt = COCO(self.ann_file)
        self.cat_ids = cocoGt.getCatIds(catNms=self.classes)
        self.img_ids = cocoGt.getImgIds()

        result_files, tmp_dir = self.format_results(self.results, self.img_ids)

        eval_results = OrderedDict()
        for metric in self.metrics:
            iou_type = metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                cocoDt = cocoGt.loadRes(predictions)
            except IndexError:
                print('The testing results of the whole dataset is empty.')
                break

            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(self.proposal_nums)
            cocoEval.params.iouThrs = self.iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }

            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            metric_items = [
                'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
            ]

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = float(
                    f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}')
                eval_results[key] = val
            ap = cocoEval.stats[:6]
            eval_results[f'{metric}_mAP_copypaste'] = (
                f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    def merge(self, other: 'ImageInstanceSegmentationCOCOMetric'):
        self.results.extend(other.results)

    def __getstate__(self):
        return self.results

    def __setstate__(self, state):
        self.__init__()
        self.results = state

    def format_results(self, results, img_ids, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            data_infos(list[tuple | numpy.ndarray]): data information
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(img_ids), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(img_ids)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self.img_ids)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self.img_ids)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                # Here we skip invalid predicted labels, as we use the fixed num_classes of 80 (COCO)
                # (assuming the num class of input dataset is no more than 80).
                # Recommended manually set `num_classes=${your test dataset class num}` in the
                # configuration.json in practice.
                if label >= len(self.classes):
                    break
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self.img_ids)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # Here we skip invalid predicted labels, as we use the fixed num_classes of 80 (COCO)
                # (assuming the num class of input dataset is no more than 80).
                # Recommended manually set `num_classes=${your test dataset class num}` in the
                # configuration.json in practice.
                if label >= len(self.classes):
                    break
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            dump(json_results[0], result_files['bbox'])
            dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files


def encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code.

    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).

    Returns:
        list | tuple: RLE encoded mask.
    """
    if isinstance(mask_results, tuple):  # mask scoring
        cls_segms, cls_mask_scores = mask_results
    else:
        cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = [[] for _ in range(num_classes)]
    for i in range(len(cls_segms)):
        for cls_segm in cls_segms[i]:
            encoded_mask_results[i].append(
                mask_util.encode(
                    np.array(
                        cls_segm[:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    if isinstance(mask_results, tuple):
        return encoded_mask_results, cls_mask_scores
    else:
        return encoded_mask_results
