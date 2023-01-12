# Copyright (c) Alibaba, Inc. and its affiliates.

import os

import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from modelscope.outputs import OutputKeys
from modelscope.preprocessors.image import load_image
from modelscope.utils import logger as logging

logger = logging.get_logger()


def numpy_to_cv2img(img_array):
    """to convert a np.array with shape(h, w) to cv2 img

    Args:
        img_array (np.array): input data

    Returns:
        cv2 img
    """
    img_array = (img_array - img_array.min()) / (
        img_array.max() - img_array.min() + 1e-5)
    img_array = (img_array * 255).astype(np.uint8)
    img_array = cv2.applyColorMap(img_array, cv2.COLORMAP_JET)
    return img_array


def draw_joints(image, np_kps, score, threshold=0.2):
    lst_parent_ids_17 = [0, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14]
    lst_left_ids_17 = [1, 3, 5, 7, 9, 11, 13, 15]
    lst_right_ids_17 = [2, 4, 6, 8, 10, 12, 14, 16]

    lst_parent_ids_15 = [0, 0, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 1]
    lst_left_ids_15 = [2, 3, 4, 8, 9, 10]
    lst_right_ids_15 = [5, 6, 7, 11, 12, 13]

    if np_kps.shape[0] == 17:
        lst_parent_ids = lst_parent_ids_17
        lst_left_ids = lst_left_ids_17
        lst_right_ids = lst_right_ids_17

    elif np_kps.shape[0] == 15:
        lst_parent_ids = lst_parent_ids_15
        lst_left_ids = lst_left_ids_15
        lst_right_ids = lst_right_ids_15

    for i in range(len(lst_parent_ids)):
        pid = lst_parent_ids[i]
        if i == pid:
            continue

        if (score[i] < threshold or score[1] < threshold):
            continue

        if i in lst_left_ids and pid in lst_left_ids:
            color = (0, 255, 0)
        elif i in lst_right_ids and pid in lst_right_ids:
            color = (255, 0, 0)
        else:
            color = (0, 255, 255)

        cv2.line(image, (int(np_kps[i, 0]), int(np_kps[i, 1])),
                 (int(np_kps[pid][0]), int(np_kps[pid, 1])), color, 3)

    for i in range(np_kps.shape[0]):
        if score[i] < threshold:
            continue
        cv2.circle(image, (int(np_kps[i, 0]), int(np_kps[i, 1])), 5,
                   (0, 0, 255), -1)


def draw_box(image, box):
    cv2.rectangle(image, (int(box[0]), int(box[1])),
                  (int(box[2]), int(box[3])), (0, 0, 255), 2)


def realtime_object_detection_bbox_vis(image, bboxes):
    for bbox in bboxes:
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      (255, 0, 0), 2)
    return image


def draw_keypoints(output, original_image):
    poses = np.array(output[OutputKeys.KEYPOINTS])
    scores = np.array(output[OutputKeys.SCORES])
    boxes = np.array(output[OutputKeys.BOXES])
    assert len(poses) == len(scores) and len(poses) == len(boxes)
    image = cv2.imread(original_image, -1)
    for i in range(len(poses)):
        draw_box(image, np.array(boxes[i]))
        draw_joints(image, np.array(poses[i]), np.array(scores[i]))
    return image


def draw_106face_keypoints(in_path,
                           keypoints,
                           boxes,
                           scale=4.0,
                           save_path=None):
    face_contour_point_index = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
    ]
    left_eye_brow_point_index = [33, 34, 35, 36, 37, 38, 39, 40, 41, 33]
    right_eye_brow_point_index = [42, 43, 44, 45, 46, 47, 48, 49, 50, 42]
    left_eye_point_index = [66, 67, 68, 69, 70, 71, 72, 73, 66]
    right_eye_point_index = [75, 76, 77, 78, 79, 80, 81, 82, 75]
    nose_bridge_point_index = [51, 52, 53, 54]
    nose_contour_point_index = [55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
    mouth_outer_point_index = [
        84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 84
    ]
    mouth_inter_point_index = [96, 97, 98, 99, 100, 101, 102, 103, 96]

    img = cv2.imread(in_path)

    for i in range(len(boxes)):
        draw_box(img, np.array(boxes[i]))

    image = cv2.resize(img, dsize=None, fx=scale, fy=scale)

    def draw_line(point_index, image, point):
        for i in range(len(point_index) - 1):
            cur_index = point_index[i]
            next_index = point_index[i + 1]
            cur_pt = (int(point[cur_index][0] * scale),
                      int(point[cur_index][1] * scale))
            next_pt = (int(point[next_index][0] * scale),
                       int(point[next_index][1] * scale))
            cv2.line(image, cur_pt, next_pt, (0, 0, 255), thickness=2)

    for i in range(len(keypoints)):
        points = keypoints[i]

        draw_line(face_contour_point_index, image, points)
        draw_line(left_eye_brow_point_index, image, points)
        draw_line(right_eye_brow_point_index, image, points)
        draw_line(left_eye_point_index, image, points)
        draw_line(right_eye_point_index, image, points)
        draw_line(nose_bridge_point_index, image, points)
        draw_line(nose_contour_point_index, image, points)
        draw_line(mouth_outer_point_index, image, points)
        draw_line(mouth_inter_point_index, image, points)

        size = len(points)
        for i in range(size):
            x = int(points[i][0])
            y = int(points[i][1])
            cv2.putText(image, str(i), (int(x * scale), int(y * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.circle(image, (int(x * scale), int(y * scale)), 2, (0, 255, 0),
                       cv2.FILLED)

    if save_path is not None:
        cv2.imwrite(save_path, image)

    return image


def draw_face_detection_no_lm_result(img_path, detection_result):
    bboxes = np.array(detection_result[OutputKeys.BOXES])
    scores = np.array(detection_result[OutputKeys.SCORES])
    img = cv2.imread(img_path)
    assert img is not None, f"Can't read img: {img_path}"
    for i in range(len(scores)):
        bbox = bboxes[i].astype(np.int32)
        x1, y1, x2, y2 = bbox
        score = scores[i]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            img,
            f'{score:.2f}', (x1, y2),
            1,
            1.0, (0, 255, 0),
            thickness=1,
            lineType=8)
    print(f'Found {len(scores)} faces')
    return img


def draw_facial_expression_result(img_path, facial_expression_result):
    scores = facial_expression_result[OutputKeys.SCORES]
    labels = facial_expression_result[OutputKeys.LABELS]
    label = labels[np.argmax(scores)]
    img = cv2.imread(img_path)
    assert img is not None, f"Can't read img: {img_path}"
    cv2.putText(
        img,
        'facial expression: {}'.format(label), (10, 10),
        1,
        1.0, (0, 255, 0),
        thickness=1,
        lineType=8)
    print('facial expression: {}'.format(label))
    return img


def draw_face_attribute_result(img_path, face_attribute_result):
    scores = face_attribute_result[OutputKeys.SCORES]
    labels = face_attribute_result[OutputKeys.LABELS]
    label_gender = labels[0][np.argmax(scores[0])]
    label_age = labels[1][np.argmax(scores[1])]
    img = cv2.imread(img_path)
    assert img is not None, f"Can't read img: {img_path}"
    cv2.putText(
        img,
        'face gender: {}'.format(label_gender), (10, 10),
        1,
        1.0, (0, 255, 0),
        thickness=1,
        lineType=8)

    cv2.putText(
        img,
        'face age interval: {}'.format(label_age), (10, 40),
        1,
        1.0, (255, 0, 0),
        thickness=1,
        lineType=8)
    logger.info('face gender: {}'.format(label_gender))
    logger.info('face age interval: {}'.format(label_age))
    return img


def draw_face_detection_result(img_path, detection_result):
    bboxes = np.array(detection_result[OutputKeys.BOXES])
    kpss = np.array(detection_result[OutputKeys.KEYPOINTS])
    scores = np.array(detection_result[OutputKeys.SCORES])
    img = cv2.imread(img_path)
    assert img is not None, f"Can't read img: {img_path}"
    for i in range(len(scores)):
        bbox = bboxes[i].astype(np.int32)
        kps = kpss[i].reshape(-1, 2).astype(np.int32)
        score = scores[i]
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for kp in kps:
            cv2.circle(img, tuple(kp), 1, (0, 0, 255), 1)
        cv2.putText(
            img,
            f'{score:.2f}', (x1, y2),
            1,
            1.0, (0, 255, 0),
            thickness=1,
            lineType=8)
    print(f'Found {len(scores)} faces')
    return img


def draw_card_detection_result(img_path, detection_result):

    def warp_img(src_img, kps, ratio):
        short_size = 500
        if ratio > 1:
            obj_h = short_size
            obj_w = int(obj_h * ratio)
        else:
            obj_w = short_size
            obj_h = int(obj_w / ratio)
        input_pts = np.float32([kps[0], kps[1], kps[2], kps[3]])
        output_pts = np.float32([[0, obj_h - 1], [0, 0], [obj_w - 1, 0],
                                 [obj_w - 1, obj_h - 1]])
        M = cv2.getPerspectiveTransform(input_pts, output_pts)
        obj_img = cv2.warpPerspective(src_img, M, (obj_w, obj_h))
        return obj_img

    bboxes = np.array(detection_result[OutputKeys.BOXES])
    kpss = np.array(detection_result[OutputKeys.KEYPOINTS])
    scores = np.array(detection_result[OutputKeys.SCORES])
    img_list = []
    ver_col = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    img = cv2.imread(img_path)
    img_list += [img]
    assert img is not None, f"Can't read img: {img_path}"
    for i in range(len(scores)):
        bbox = bboxes[i].astype(np.int32)
        kps = kpss[i].reshape(-1, 2).astype(np.int32)
        _w = (kps[0][0] - kps[3][0])**2 + (kps[0][1] - kps[3][1])**2
        _h = (kps[0][0] - kps[1][0])**2 + (kps[0][1] - kps[1][1])**2
        ratio = 1.59 if _w >= _h else 1 / 1.59
        card_img = warp_img(img, kps, ratio)
        img_list += [card_img]
        score = scores[i]
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 4)
        for k, kp in enumerate(kps):
            cv2.circle(img, tuple(kp), 1, color=ver_col[k], thickness=10)
        cv2.putText(
            img,
            f'{score:.2f}', (x1, y2),
            1,
            1.0, (0, 255, 0),
            thickness=1,
            lineType=8)
    return img_list


def created_boxed_image(image_in, box):
    image = load_image(image_in)
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                  (0, 255, 0), 3)
    return img


def show_video_tracking_result(video_in_path, bboxes, video_save_path):
    cap = cv2.VideoCapture(video_in_path)
    for i in range(len(bboxes)):
        box = bboxes[i]
        success, frame = cap.read()
        if success is False:
            raise Exception(video_in_path,
                            ' can not be correctly decoded by OpenCV.')
        if i == 0:
            size = (frame.shape[1], frame.shape[0])
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            video_writer = cv2.VideoWriter(video_save_path, fourcc,
                                           cap.get(cv2.CAP_PROP_FPS), size,
                                           True)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0),
                      5)
        video_writer.write(frame)
    video_writer.release
    cap.release()


def show_video_object_detection_result(video_in_path, bboxes_list, labels_list,
                                       video_save_path):

    PALETTE = {
        'person': [128, 0, 0],
        'bicycle': [128, 128, 0],
        'car': [64, 0, 0],
        'motorcycle': [0, 128, 128],
        'bus': [64, 128, 0],
        'truck': [192, 128, 0],
        'traffic light': [64, 0, 128],
        'stop sign': [192, 0, 128],
    }
    from tqdm import tqdm
    import math
    cap = cv2.VideoCapture(video_in_path)
    with tqdm(total=len(bboxes_list)) as pbar:
        pbar.set_description(
            'Writing results to video: {}'.format(video_save_path))
        for i in range(len(bboxes_list)):
            bboxes = bboxes_list[i].astype(int)
            labels = labels_list[i]
            success, frame = cap.read()
            if success is False:
                raise Exception(video_in_path,
                                ' can not be correctly decoded by OpenCV.')
            if i == 0:
                size = (frame.shape[1], frame.shape[0])
                fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                video_writer = cv2.VideoWriter(video_save_path, fourcc,
                                               cap.get(cv2.CAP_PROP_FPS), size,
                                               True)

            FONT_SCALE = 1e-3  # Adjust for larger font size in all images
            THICKNESS_SCALE = 1e-3  # Adjust for larger thickness in all images
            TEXT_Y_OFFSET_SCALE = 1e-2  # Adjust for larger Y-offset of text and bounding box
            H, W, _ = frame.shape
            zeros_mask = np.zeros((frame.shape)).astype(np.uint8)
            for bbox, l in zip(bboxes, labels):
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                              PALETTE[l], 1)
                cv2.putText(
                    frame,
                    l, (bbox[0], bbox[1] - int(TEXT_Y_OFFSET_SCALE * H)),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=min(H, W) * FONT_SCALE,
                    thickness=math.ceil(min(H, W) * THICKNESS_SCALE),
                    color=PALETTE[l])
                zeros_mask = cv2.rectangle(
                    zeros_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                    color=PALETTE[l],
                    thickness=-1)

            frame = cv2.addWeighted(frame, 1., zeros_mask, .65, 0)
            video_writer.write(frame)
            pbar.update(1)
    video_writer.release
    cap.release()


def panoptic_seg_masks_to_image(masks):
    draw_img = np.zeros([masks[0].shape[0], masks[0].shape[1], 3])
    from mmdet.core.visualization.palette import get_palette
    mask_palette = get_palette('coco', 133)

    from mmdet.core.visualization.image import _get_bias_color
    taken_colors = set([0, 0, 0])
    for i, mask in enumerate(masks):
        color_mask = mask_palette[i]
        while tuple(color_mask) in taken_colors:
            color_mask = _get_bias_color(color_mask)
        taken_colors.add(tuple(color_mask))

        mask = mask.astype(bool)
        draw_img[mask] = color_mask

    return draw_img


def semantic_seg_masks_to_image(masks):
    from mmdet.core.visualization.palette import get_palette
    mask_palette = get_palette('coco', 133)

    draw_img = np.zeros([masks[0].shape[0], masks[0].shape[1], 3])

    for i, mask in enumerate(masks):
        color_mask = mask_palette[i]
        mask = mask.astype(bool)
        draw_img[mask] = color_mask
    return draw_img


def show_video_summarization_result(video_in_path, result, video_save_path):
    frame_indexes = result[OutputKeys.OUTPUT]
    cap = cv2.VideoCapture(video_in_path)
    for i in range(len(frame_indexes)):
        idx = frame_indexes[i]
        success, frame = cap.read()
        if success is False:
            raise Exception(video_in_path,
                            ' can not be correctly decoded by OpenCV.')
        if i == 0:
            size = (frame.shape[1], frame.shape[0])
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            video_writer = cv2.VideoWriter(video_save_path, fourcc,
                                           cap.get(cv2.CAP_PROP_FPS), size,
                                           True)
        if idx == 1:
            video_writer.write(frame)
    video_writer.release()
    cap.release()


def show_image_object_detection_auto_result(img_path,
                                            detection_result,
                                            save_path=None):
    scores = detection_result[OutputKeys.SCORES]
    labels = detection_result[OutputKeys.LABELS]
    bboxes = detection_result[OutputKeys.BOXES]
    img = cv2.imread(img_path)
    assert img is not None, f"Can't read img: {img_path}"

    for (score, label, box) in zip(scores, labels, bboxes):
        cv2.rectangle(img, (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])), (0, 0, 255), 2)
        cv2.putText(
            img,
            f'{score:.2f}', (int(box[0]), int(box[1])),
            1,
            1.0, (0, 255, 0),
            thickness=1,
            lineType=8)
        cv2.putText(
            img,
            label, (int(box[0]), int(box[3])),
            1,
            1.0, (0, 255, 0),
            thickness=1,
            lineType=8)

    if save_path is not None:
        cv2.imwrite(save_path, img)
    return img


def depth_to_color(depth):
    colormap = plt.get_cmap('plasma')
    depth_color = (colormap(
        (depth.max() - depth) / depth.max()) * 2**8).astype(np.uint8)[:, :, :3]
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)
    return depth_color


def show_video_depth_estimation_result(depths, video_save_path):
    height, width, layers = depths[0].shape
    out = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'MP4V'), 25,
                          (width, height))
    for (i, img) in enumerate(depths):
        out.write(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
    out.release()


def masks_visualization(masks, palette):
    vis_masks = []
    for f in range(masks.shape[0]):
        img_E = Image.fromarray(masks[f])
        img_E.putpalette(palette)
        vis_masks.append(img_E)
    return vis_masks


# This implementation is adopted from LoFTR,
# made public available under the Apache License, Version 2.0,
# at https://github.com/zju3dv/LoFTR


def make_matching_figure(img0,
                         img1,
                         mkpts0,
                         mkpts1,
                         color,
                         kpts0=None,
                         kpts1=None,
                         text=[],
                         dpi=75,
                         path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[
        0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [
            matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                    (fkpts0[i, 1], fkpts1[i, 1]),
                                    transform=fig.transFigure,
                                    c=color[i],
                                    linewidth=1) for i in range(len(mkpts0))
        ]

        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01,
        0.99,
        '\n'.join(text),
        transform=fig.axes[0].transAxes,
        fontsize=15,
        va='top',
        ha='left',
        color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig


def match_pair_visualization(img_name0,
                             img_name1,
                             kpts0,
                             kpts1,
                             conf,
                             output_filename='quadtree_match.png',
                             method='QuadTreeAttention'):

    print(f'Found {len(kpts0)} matches')

    # visualize the matches
    img0 = cv2.imread(str(img_name0))
    img1 = cv2.imread(str(img_name1))

    # Draw
    color = cm.jet(conf)
    text = [
        method,
        'Matches: {}'.format(len(kpts0)),
    ]
    fig = make_matching_figure(img0, img1, kpts0, kpts1, color, text=text)

    # save the figure
    fig.savefig(str(output_filename), dpi=300, bbox_inches='tight')
