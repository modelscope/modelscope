# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
import logging
import os.path as osp
from collections import OrderedDict

import numpy as np
import pandas as pd
from detectron2.evaluation import DatasetEvaluator
from detectron2.evaluation.pascal_voc_evaluation import voc_ap
from detectron2.structures.boxes import Boxes, pairwise_iou
from detectron2.utils import comm
from scipy import interpolate


class DetEvaluator(DatasetEvaluator):

    def __init__(self, class_names, output_dir, distributed=False):
        self.num_classes = len(class_names)
        self.class_names = class_names
        self.output_dir = output_dir
        self.distributed = distributed
        self.predictions = []
        self.gts = []

    def reset(self):
        self.predictions.clear()
        self.gts.clear()

    def process(self, input, output):
        """

        :param input: dataloader
        :param output: model(input)
        :return:
        """
        gt_instances = [x['instances'].to('cpu') for x in input]
        pred_instances = [x['instances'].to('cpu') for x in output]
        self.gts.extend(gt_instances)
        self.predictions.extend(pred_instances)

    def get_instance_by_class(self, instances, c):
        instances = copy.deepcopy(instances)
        name = 'gt_classes' if instances.has('gt_classes') else 'pred_classes'
        idxs = np.where(instances.get(name).numpy() == c)[0].tolist()
        data = {}
        for k, v in instances.get_fields().items():
            data[k] = [v[i] for i in idxs]
        return data

    def evaluate(self):
        if self.distributed:
            comm.synchronize()
            self.predictions = sum(comm.gather(self.predictions, dst=0), [])
            self.gts = sum(comm.gather(self.gts, dst=0), [])
            if not comm.is_main_process():
                return
        logger = logging.getLogger('detectron2.human.' + __name__)
        logger.info(', '.join([f'{a}' for a in self.class_names]))
        maps = []
        precisions = []
        recalls = []
        for iou_th in [0.3, 0.5, 0.7]:
            aps, prs, ths = self.calc_map(iou_th)
            map = np.nanmean([x for x in aps if x > 0.01])
            maps.append(map)
            logger.info(f'iou_th:{iou_th},' + 'Aps:'
                        + ','.join([f'{ap:.2f}'
                                    for ap in aps]) + f', {map:.3f}')
            precision, recall = zip(*prs)
            logger.info('precision:'
                        + ', '.join([f'{p:.2f}' for p in precision]))
            logger.info('recall:   ' + ', '.join([f'{p:.2f}' for p in recall]))
            logger.info('score th: ' + ', '.join([f'{p:.2f}' for p in ths]))
            logger.info(f'mean-precision:{np.nanmean(precision):.3f}')
            logger.info(f'mean-recall:{np.nanmean(recall):.3f}')
            precisions.append(np.nanmean(precision))
            recalls.append(np.nanmean(recall))

        res = OrderedDict({
            'det': {
                'mAP': np.nanmean(maps),
                'precision': np.nanmean(precisions),
                'recall': np.nanmean(recalls)
            }
        })
        return res

    def calc_map(self, iou_th):
        aps = []
        prs = []
        ths = []
        # 对每个类别
        interpolate_precs = []
        for c in range(self.num_classes):
            ap, recalls, precisions, scores = self.det_eval(iou_th, c)
            if iou_th == 0.3:
                p1 = interpolate_precision(recalls, precisions)
                interpolate_precs.append(p1)
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))
            scores = np.concatenate(([1.0], scores, [0.0]))
            t = precisions + recalls
            t[t == 0] = 1e-5
            f_score = 2 * precisions * recalls / t
            f_score[np.isnan(f_score)] = 0
            idx = np.argmax(f_score)
            # print(iou_th,c,np.argmax(f_score),np.argmax(t))
            precision_recall = (precisions[idx], recalls[idx])
            prs.append(precision_recall)
            aps.append(ap)
            ths.append(scores[idx])
        if iou_th == 0.3:
            interpolate_precs = np.stack(interpolate_precs, axis=1)
            df = pd.DataFrame(data=interpolate_precs)
            df.to_csv(
                osp.join(self.output_dir, 'pr_data.csv'),
                index=False,
                columns=None)
        return aps, prs, ths

    def det_eval(self, iou_th, class_id):
        c = class_id
        class_res_gt = {}
        npos = 0
        # 对每个样本
        for i, (gt, pred) in enumerate(zip(self.gts, self.predictions)):
            gt_classes = gt.gt_classes.tolist()
            pred_classes = pred.pred_classes.tolist()
            if c not in gt_classes + pred_classes:
                continue
            pred_data = self.get_instance_by_class(pred, c)
            gt_data = self.get_instance_by_class(gt, c)
            res = {}
            if c in gt_classes:
                res.update({
                    'gt_bbox': Boxes.cat(gt_data['gt_boxes']),
                    'det': [False] * len(gt_data['gt_classes'])
                })
            if c in pred_classes:
                res.update({'pred_bbox': Boxes.cat(pred_data['pred_boxes'])})
                res.update(
                    {'pred_score': [s.item() for s in pred_data['scores']]})
            class_res_gt[i] = res
            npos += len(gt_data['gt_classes'])

        all_preds = []
        for img_id, res in class_res_gt.items():
            if 'pred_bbox' in res:
                for i in range(len(res['pred_bbox'])):
                    bbox = res['pred_bbox'][i]
                    score = res['pred_score'][i]
                    all_preds.append([img_id, bbox, score])
        sorted_preds = list(
            sorted(all_preds, key=lambda x: x[2], reverse=True))
        scores = [s[-1] for s in sorted_preds]
        nd = len(sorted_preds)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            img_id, pred_bbox, score = sorted_preds[d]
            R = class_res_gt[sorted_preds[d][0]]
            ovmax = -np.inf
            if 'gt_bbox' in R:
                gt_bbox = R['gt_bbox']
                IoUs = pairwise_iou(pred_bbox, gt_bbox).numpy()
                ovmax = IoUs[0].max()
                jmax = np.argmax(IoUs[0])  # hit该图像的第几个gt
            if ovmax > iou_th:
                if not R['det'][jmax]:  # 该gt还没有预测过
                    tp[d] = 1.0
                    R['det'][jmax] = True
                else:  # 重复预测
                    fp[d] = 1.0
            else:
                fp[d] = 1.0
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, False)
        return ap, rec, prec, scores


def interpolate_precision(rec, prec):
    rec = np.concatenate(([0.0], rec, [1.0, 1.1]))
    prec = np.concatenate(([1.0], prec, [0.0]))
    for i in range(prec.size - 1, 0, -1):
        prec[i - 1] = np.maximum(prec[i - 1], prec[i])
    i = np.where(rec[1:] != rec[:-1])[0]  # 从recall改变的地方取值
    rec, prec = rec[i], prec[i]
    f = interpolate.interp1d(rec, prec)
    r1 = np.linspace(0, 1, 101)
    p1 = f(r1)
    return p1
