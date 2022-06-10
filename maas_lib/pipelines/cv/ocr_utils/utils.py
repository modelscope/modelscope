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
    return (point_dist(box[0], box[1], box[2], box[3]) 
            + point_dist(box[4], box[5], box[6], box[7])) / 2


def point_dist(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))


def draw_polygons(img, polygons):
    print(polygons.tolist())
    for p in polygons.tolist():
        p = [int(o) for o in p]
        cv2.line(img, (p[0], p[1]), (p[2], p[3]), (0, 255, 0), 1)
        cv2.line(img, (p[2], p[3]), (p[4], p[5]), (0, 255, 0), 1)
        cv2.line(img, (p[4], p[5]), (p[6], p[7]), (0, 255, 0), 1)
        cv2.line(img, (p[6], p[7]), (p[0], p[1]), (0, 255, 0), 1)
    return img
