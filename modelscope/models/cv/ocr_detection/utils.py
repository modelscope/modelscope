# Copyright (c) Alibaba, Inc. and its affiliates.
import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


def rboxes_to_polygons(rboxes):
    """
    Convert rboxes to polygons
    ARGS
        `rboxes`: [n, 5]
    RETURN
        `polygons`: [n, 8]
    """

    theta = rboxes[:, 4:5]
    cxcy = rboxes[:, :2]
    half_w = rboxes[:, 2:3] / 2.
    half_h = rboxes[:, 3:4] / 2.
    v1 = np.hstack([np.cos(theta) * half_w, np.sin(theta) * half_w])
    v2 = np.hstack([-np.sin(theta) * half_h, np.cos(theta) * half_h])
    p1 = cxcy - v1 - v2
    p2 = cxcy + v1 - v2
    p3 = cxcy + v1 + v2
    p4 = cxcy - v1 + v2
    polygons = np.hstack([p1, p2, p3, p4])
    return polygons


def cal_width(box):
    pd1 = point_dist(box[0], box[1], box[2], box[3])
    pd2 = point_dist(box[4], box[5], box[6], box[7])
    return (pd1 + pd2) / 2


def point_dist(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))


def draw_polygons(img, polygons):
    for p in polygons.tolist():
        p = [int(o) for o in p]
        cv2.line(img, (p[0], p[1]), (p[2], p[3]), (0, 255, 0), 1)
        cv2.line(img, (p[2], p[3]), (p[4], p[5]), (0, 255, 0), 1)
        cv2.line(img, (p[4], p[5]), (p[6], p[7]), (0, 255, 0), 1)
        cv2.line(img, (p[6], p[7]), (p[0], p[1]), (0, 255, 0), 1)
    return img


def nms_python(boxes):
    boxes = sorted(boxes, key=lambda x: -x[8])
    nms_flag = [True] * len(boxes)
    for i, a in enumerate(boxes):
        if not nms_flag[i]:
            continue
        else:
            for j, b in enumerate(boxes):
                if not j > i:
                    continue
                if not nms_flag[j]:
                    continue
                score_a = a[8]
                score_b = b[8]
                rbox_a = polygon2rbox(a[:8])
                rbox_b = polygon2rbox(b[:8])
                if point_in_rbox(rbox_a[:2], rbox_b) or point_in_rbox(
                        rbox_b[:2], rbox_a):
                    if score_a > score_b:
                        nms_flag[j] = False
    boxes_nms = []
    for i, box in enumerate(boxes):
        if nms_flag[i]:
            boxes_nms.append(box)
    return boxes_nms


def point_in_rbox(c, rbox):
    cx0, cy0 = c[0], c[1]
    cx1, cy1 = rbox[0], rbox[1]
    w, h = rbox[2], rbox[3]
    theta = rbox[4]
    dist_x = np.abs((cx1 - cx0) * np.cos(theta) + (cy1 - cy0) * np.sin(theta))
    dist_y = np.abs(-(cx1 - cx0) * np.sin(theta) + (cy1 - cy0) * np.cos(theta))
    return ((dist_x < w / 2.0) and (dist_y < h / 2.0))


def polygon2rbox(polygon):
    x1, x2, x3, x4 = polygon[0], polygon[2], polygon[4], polygon[6]
    y1, y2, y3, y4 = polygon[1], polygon[3], polygon[5], polygon[7]
    c_x = (x1 + x2 + x3 + x4) / 4
    c_y = (y1 + y2 + y3 + y4) / 4
    w1 = point_dist(x1, y1, x2, y2)
    w2 = point_dist(x3, y3, x4, y4)
    h1 = point_line_dist(c_x, c_y, x1, y1, x2, y2)
    h2 = point_line_dist(c_x, c_y, x3, y3, x4, y4)
    h = h1 + h2
    w = (w1 + w2) / 2
    theta1 = np.arctan2(y2 - y1, x2 - x1)
    theta2 = np.arctan2(y3 - y4, x3 - x4)
    theta = (theta1 + theta2) / 2.0
    return [c_x, c_y, w, h, theta]


def point_line_dist(px, py, x1, y1, x2, y2):
    eps = 1e-6
    dx = x2 - x1
    dy = y2 - y1
    div = np.sqrt(dx * dx + dy * dy) + eps
    dist = np.abs(px * dy - py * dx + x2 * y1 - y2 * x1) / div
    return dist


# Part of the implementation is borrowed and modified from DB,
# publicly available at https://github.com/MhLiao/DB.
def polygons_from_bitmap(pred, _bitmap, dest_width, dest_height):
    """
    _bitmap: single map with shape (1, H, W),
        whose values are binarized as {0, 1}
    """

    assert _bitmap.size(0) == 1
    bitmap = _bitmap.cpu().numpy()[0]
    pred = pred.cpu().detach().numpy()[0]
    height, width = bitmap.shape
    boxes = []
    scores = []

    contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8),
                                   cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours[:100]:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape((-1, 2))
        if points.shape[0] < 4:
            continue

        score = box_score_fast(pred, points.reshape(-1, 2))
        if 0.7 > score:
            continue

        if points.shape[0] > 2:
            box = unclip(points, unclip_ratio=2.0)
            if len(box) > 1:
                continue
        else:
            continue
        box = box.reshape(-1, 2)
        _, sside = get_mini_boxes(box.reshape((-1, 1, 2)))
        if sside < 3 + 2:
            continue

        if not isinstance(dest_width, int):
            dest_width = dest_width.item()
            dest_height = dest_height.item()

        box[:, 0] = np.clip(
            np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(
            np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes.append(box.tolist())
        scores.append(score)
    return boxes, scores


def boxes_from_bitmap(pred, _bitmap, dest_width, dest_height, is_numpy=False):
    """
    _bitmap: single map with shape (1, H, W),
        whose values are binarized as {0, 1}
    """
    if is_numpy:
        bitmap = _bitmap[0]
        pred = pred[0]
    else:
        bitmap = _bitmap.cpu().numpy()[0]
        pred = pred.cpu().detach().numpy()[0]
    height, width = bitmap.shape
    boxes = []
    scores = []

    contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8),
                                   cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours[:1000]:
        points, sside = get_mini_boxes(contour)
        if sside < 3:
            continue
        points = np.array(points)

        score = box_score_fast(pred, points.reshape(-1, 2))
        if 0.3 > score:
            continue

        box = unclip(points, unclip_ratio=1.5).reshape(-1, 1, 2)
        box, sside = get_mini_boxes(box)

        if sside < 3 + 2:
            continue

        box = np.array(box).astype(np.int32)
        if not isinstance(dest_width, int):
            dest_width = dest_width.item()
            dest_height = dest_height.item()

        box[:, 0] = np.clip(
            np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(
            np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes.append(box.reshape(-1).tolist())
        scores.append(score)
    return boxes, scores


def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, min(bounding_box[1])
