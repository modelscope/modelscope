# Part of the implementation is borrowed and modified from MMDetection, publicly available at
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/visualization/image.py
import itertools

import cv2
import numpy as np
import pycocotools.mask as maskUtils
import torch

from modelscope.outputs import OutputKeys


def get_seg_bboxes(bboxes, labels, segms=None, class_names=None, score_thr=0.):
    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes.shape[0] == labels.shape[0], \
        'bboxes.shape[0] and labels.shape[0] should have the same length.'
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    bboxes_names = []
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        label_name = class_names[
            label] if class_names is not None else f'class {label}'
        bbox = [0 if b < 0 else b for b in list(bbox)]
        bbox.append(label_name)
        bbox.append(segms[i].astype(bool))
        bboxes_names.append(bbox)

    return bboxes_names


def get_img_seg_results(det_rawdata=None,
                        class_names=None,
                        score_thr=0.3,
                        is_decode=True):
    '''
       Get all boxes of one image.
       score_thr: Classification probability threshold。
       output format: [ [x1,y1,x2,y2, prob, cls_name, mask], [x1,y1,x2,y2, prob, cls_name, mask], ... ]
    '''
    assert det_rawdata is not None, 'det_rawdata should be not None.'
    assert class_names is not None, 'class_names should be not None.'

    if isinstance(det_rawdata, tuple):
        bbox_result, segm_result = det_rawdata
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = det_rawdata, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = list(itertools.chain(*segm_result))
        if is_decode:
            segms = maskUtils.decode(segms)
            segms = segms.transpose(2, 0, 1)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)

    bboxes_names = get_seg_bboxes(
        bboxes,
        labels,
        segms=segms,
        class_names=class_names,
        score_thr=score_thr)

    return bboxes_names


def get_img_ins_seg_result(img_seg_result=None,
                           class_names=None,
                           score_thr=0.3):
    assert img_seg_result is not None, 'img_seg_result should be not None.'
    assert class_names is not None, 'class_names should be not None.'

    img_seg_result = get_img_seg_results(
        det_rawdata=(img_seg_result[0], img_seg_result[1]),
        class_names=class_names,
        score_thr=score_thr,
        is_decode=False)

    results_dict = {
        OutputKeys.BOXES: [],
        OutputKeys.MASKS: [],
        OutputKeys.LABELS: [],
        OutputKeys.SCORES: []
    }
    for seg_result in img_seg_result:

        box = [
            np.int(seg_result[0]),
            np.int(seg_result[1]),
            np.int(seg_result[2]),
            np.int(seg_result[3])
        ]
        score = np.float(seg_result[4])
        category = seg_result[5]

        mask = np.array(seg_result[6], order='F', dtype='uint8')
        mask = mask.astype(np.float)

        results_dict[OutputKeys.BOXES].append(box)
        results_dict[OutputKeys.MASKS].append(mask)
        results_dict[OutputKeys.SCORES].append(score)
        results_dict[OutputKeys.LABELS].append(category)

    return results_dict


def show_result(
    img,
    result,
    out_file='result.jpg',
    show_box=True,
    show_label=True,
    show_score=True,
    alpha=0.5,
    fontScale=0.5,
    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
    thickness=1,
):

    assert isinstance(img, (str, np.ndarray)), \
        f'img must be str or np.ndarray, but got {type(img)}.'

    if isinstance(img, str):
        img = cv2.imread(img)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = img.astype(np.float32)

    labels = result[OutputKeys.LABELS]
    scores = result[OutputKeys.SCORES]
    boxes = result[OutputKeys.BOXES]
    masks = result[OutputKeys.MASKS]

    for label, score, box, mask in zip(labels, scores, boxes, masks):

        random_color = np.array([
            np.random.random() * 255.0,
            np.random.random() * 255.0,
            np.random.random() * 255.0
        ])

        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        if show_box:
            cv2.rectangle(
                img, (x1, y1), (x2, y2), random_color, thickness=thickness)
        if show_label or show_score:
            if show_label and show_score:
                text = '{}|{}'.format(label, round(float(score), 2))
            elif show_label:
                text = '{}'.format(label)
            else:
                text = '{}'.format(round(float(score), 2))

            retval, baseLine = cv2.getTextSize(
                text,
                fontFace=fontFace,
                fontScale=fontScale,
                thickness=thickness)
            cv2.rectangle(
                img, (x1, y1 - retval[1] - baseLine), (x1 + retval[0], y1),
                thickness=-1,
                color=(0, 0, 0))
            cv2.putText(
                img,
                text, (x1, y1 - baseLine),
                fontScale=fontScale,
                fontFace=fontFace,
                thickness=thickness,
                color=random_color)

        idx = np.nonzero(mask)
        img[idx[0], idx[1], :] *= 1.0 - alpha
        img[idx[0], idx[1], :] += alpha * random_color

    cv2.imwrite(out_file, img)


def get_maskdino_ins_seg_result(maskdino_seg_result,
                                class_names,
                                score_thr=0.3):
    scores = maskdino_seg_result['scores'].detach().cpu().numpy()
    pred_masks = maskdino_seg_result['pred_masks'].detach().cpu().numpy()
    pred_boxes = maskdino_seg_result['pred_boxes'].detach().cpu().numpy()
    pred_classes = maskdino_seg_result['pred_classes'].detach().cpu().numpy()

    thresholded_idxs = np.array(scores) >= score_thr
    scores = scores[thresholded_idxs]
    pred_classes = pred_classes[thresholded_idxs]
    pred_masks = pred_masks[thresholded_idxs]
    pred_boxes = pred_boxes[thresholded_idxs]

    results_dict = {
        OutputKeys.BOXES: [],
        OutputKeys.MASKS: [],
        OutputKeys.LABELS: [],
        OutputKeys.SCORES: []
    }
    for score, cls, mask, box in zip(scores, pred_classes, pred_masks,
                                     pred_boxes):
        score = np.float64(score)
        label = class_names[int(cls)]
        mask = np.array(mask, dtype=np.float64)
        box = [
            np.int64(box[0]),
            np.int64(box[1]),
            np.int64(box[2]),
            np.int64(box[3])
        ]
        results_dict[OutputKeys.SCORES].append(score)
        results_dict[OutputKeys.LABELS].append(label)
        results_dict[OutputKeys.MASKS].append(mask)
        results_dict[OutputKeys.BOXES].append(box)

    return results_dict
