import cv2
import numpy as np


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
    return (
        point_dist(
            box[0], box[1], box[2], box[3]
        ) + point_dist(
            box[4], box[5], box[6], box[7]
        )
    ) / 2


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
    dist_y = np.abs(
        (cx1 - cx0) * (-1.0) * np.sin(theta) + (cy1 - cy0) * np.cos(theta)
    )
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
