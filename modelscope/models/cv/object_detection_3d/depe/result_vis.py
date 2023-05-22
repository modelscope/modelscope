"""
The implementation here is modified based on BEVDet, originally Apache-2.0 license and publicly available at
https://github.com/HuangJunJie2017/BEVDet/blob/dev2.0/tools/analysis_tools/vis.py
"""
import argparse
import os
import pickle

import cv2
import json
import numpy as np
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes as LB
from pyquaternion.quaternion import Quaternion


def check_point_in_img(points, height, width):
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    valid = np.logical_and(
        valid, np.logical_and(points[:, 0] < width, points[:, 1] < height))
    return valid


def depth2color(depth):
    gray = max(0, min((depth + 2.5) / 3.0, 1.0))
    max_lumi = 200
    colors = np.array(
        [[max_lumi, 0, max_lumi], [max_lumi, 0, 0], [max_lumi, max_lumi, 0],
         [0, max_lumi, 0], [0, max_lumi, max_lumi], [0, 0, max_lumi]],
        dtype=np.float32)
    if gray == 1:
        return tuple(colors[-1].tolist())
    num_rank = len(colors) - 1
    rank = np.floor(gray * num_rank).astype(int)
    diff = (gray - rank / num_rank) * num_rank
    tmp = colors[rank + 1] - colors[rank]
    return tuple((colors[rank] + tmp * diff).tolist())


def lidar2img(points_lidar, camrera_info):
    points_lidar_homogeneous = \
        np.concatenate([points_lidar,
                        np.ones((points_lidar.shape[0], 1),
                                dtype=points_lidar.dtype)], axis=1)
    camera2lidar = np.eye(4, dtype=np.float32)
    camera2lidar[:3, :3] = camrera_info['sensor2lidar_rotation']
    camera2lidar[:3, 3] = camrera_info['sensor2lidar_translation']
    lidar2camera = np.linalg.inv(camera2lidar)
    points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
    points_camera = points_camera_homogeneous[:, :3]
    valid = np.ones((points_camera.shape[0]), dtype=bool)
    valid = np.logical_and(points_camera[:, -1] > 0.5, valid)
    points_camera = points_camera / points_camera[:, 2:3]
    camera2img = camrera_info['cam_intrinsic']
    points_img = points_camera @ camera2img.T
    points_img = points_img[:, :2]
    return points_img, valid


def get_lidar2global(infos):
    lidar2ego = np.eye(4, dtype=np.float32)
    lidar2ego[:3, :3] = Quaternion(infos['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = infos['lidar2ego_translation']
    ego2global = np.eye(4, dtype=np.float32)
    ego2global[:3, :3] = Quaternion(
        infos['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = infos['ego2global_translation']
    return ego2global @ lidar2ego


def plot_result(res_path,
                vis_thred=0.3,
                version='val',
                draw_gt=True,
                save_format='image'):
    img_list = []
    # fixed parameters
    root_path = '/data/Dataset/nuScenes'
    show_range = 50  # Range of visualization in BEV
    canva_size = 1000  # Size of canva in pixel
    vis_frames = 500  # Max number of frames for visualization
    scale_factor = 2  # Trade-off between image-view and bev in size of the visualized canvas
    fps = 5  # Frame rate of video
    vis_dir = './video_result'  # Video output path
    color_map = {0: (255, 255, 0), 1: (0, 255, 255)}

    # load predicted results
    res = json.load(open(res_path, 'r'))
    # load dataset information
    info_path = os.path.join(root_path,
                             f'mmdet3d_nuscenes_30f_infos_{version}.pkl')
    with open(info_path, 'rb') as f:
        dataset = pickle.load(f)
    # prepare save path and medium
    if save_format == 'video' and not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        vout = cv2.VideoWriter(
            os.path.join(vis_dir, 'vis.mp4'), fourcc, fps,
            (int(1600 / scale_factor * 3),
             int(900 / scale_factor * 2 + canva_size)))

    draw_boxes_indexes_bev = [(0, 1), (1, 2), (2, 3), (3, 0)]
    draw_boxes_indexes_img_view = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5),
                                   (5, 6), (6, 7), (7, 4), (0, 4), (1, 5),
                                   (2, 6), (3, 7)]
    views = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ]
    dataset_dict = {}
    for sample in dataset['infos']:
        if sample['token'] not in dataset_dict:
            dataset_dict[sample['token']] = sample
    for cnt, rst_token in enumerate(res['results']):
        if cnt >= vis_frames:
            break
        # collect instances
        pred_res = res['results'][rst_token]
        infos = dataset_dict[rst_token]
        pred_boxes = [
            pred_res[rid]['translation'] + pred_res[rid]['size'] + [
                Quaternion(pred_res[rid]['rotation']).yaw_pitch_roll[0]
                + np.pi / 2
            ] for rid in range(len(pred_res))
        ]
        if len(pred_boxes) == 0:
            corners_lidar = np.zeros((0, 3), dtype=np.float32)
        else:
            pred_boxes = np.array(pred_boxes, dtype=np.float32)
            boxes = LB(pred_boxes, origin=(0.5, 0.5, 0.5))
            corners_global = boxes.corners.numpy().reshape(-1, 3)
            corners_global = np.concatenate(
                [corners_global,
                 np.ones([corners_global.shape[0], 1])],
                axis=1)
            l2g = get_lidar2global(infos)
            corners_lidar = corners_global @ np.linalg.inv(l2g).T
            corners_lidar = corners_lidar[:, :3]
        pred_flag = np.ones((corners_lidar.shape[0] // 8, ), dtype=bool)
        scores = [
            pred_res[rid]['detection_score'] for rid in range(len(pred_res))
        ]
        if draw_gt:
            gt_boxes = infos['gt_boxes']
            gt_boxes[:, -1] = gt_boxes[:, -1] + np.pi / 2
            width = gt_boxes[:, 4].copy()
            gt_boxes[:, 4] = gt_boxes[:, 3]
            gt_boxes[:, 3] = width
            corners_lidar_gt = \
                LB(infos['gt_boxes'],
                   origin=(0.5, 0.5, 0.5)).corners.numpy().reshape(-1, 3)
            corners_lidar = np.concatenate([corners_lidar, corners_lidar_gt],
                                           axis=0)
            gt_flag = np.ones((corners_lidar_gt.shape[0] // 8), dtype=bool)
            pred_flag = np.concatenate(
                [pred_flag, np.logical_not(gt_flag)], axis=0)
            scores = scores + [0 for _ in range(infos['gt_boxes'].shape[0])]
        scores = np.array(scores, dtype=np.float32)
        sort_ids = np.argsort(scores)

        # image view
        imgs = []
        for view in views:
            img = cv2.imread(infos['cams'][view]['data_path'])
            # draw instances
            corners_img, valid = lidar2img(corners_lidar, infos['cams'][view])
            valid = np.logical_and(
                valid,
                check_point_in_img(corners_img, img.shape[0], img.shape[1]))
            valid = valid.reshape(
                -1, 8)  # valid means: d>0 and visible in current view
            corners_img = corners_img.reshape(-1, 8, 2).astype(int)
            for aid in range(valid.shape[0]):
                if scores[aid] < vis_thred and pred_flag[aid]:
                    continue
                for index in draw_boxes_indexes_img_view:
                    if valid[aid, index[0]] and valid[aid, index[1]]:
                        cv2.line(
                            img,
                            corners_img[aid, index[0]],
                            corners_img[aid, index[1]],
                            color=color_map[int(pred_flag[aid])],
                            thickness=scale_factor)
            imgs.append(img)

        # bird-eye-view
        canvas = np.zeros((int(canva_size), int(canva_size), 3),
                          dtype=np.uint8)
        # draw lidar points
        lidar_points = np.fromfile(infos['lidar_path'], dtype=np.float32)
        lidar_points = lidar_points.reshape(-1, 5)[:, :3]
        lidar_points[:, 1] = -lidar_points[:, 1]
        lidar_points[:, :2] = \
            (lidar_points[:, :2] + show_range) / show_range / 2.0 * canva_size
        for p in lidar_points:
            if check_point_in_img(
                    p.reshape(1, 3), canvas.shape[1], canvas.shape[0])[0]:
                color = depth2color(p[2])
                cv2.circle(
                    canvas, (int(p[0]), int(p[1])),
                    radius=0,
                    color=color,
                    thickness=1)

        # draw instances
        corners_lidar = corners_lidar.reshape(-1, 8, 3)
        corners_lidar[:, :, 1] = -corners_lidar[:, :, 1]
        bottom_corners_bev = corners_lidar[:, [0, 3, 7, 4], :2]
        bottom_corners_bev = \
            (bottom_corners_bev + show_range) / show_range / 2.0 * canva_size
        bottom_corners_bev = np.round(bottom_corners_bev).astype(np.int32)
        center_bev = corners_lidar[:, [0, 3, 7, 4], :2].mean(axis=1)
        head_bev = corners_lidar[:, [0, 4], :2].mean(axis=1)
        canter_canvas = \
            (center_bev + show_range) / show_range / 2.0 * canva_size
        center_canvas = canter_canvas.astype(np.int32)
        head_canvas = (head_bev + show_range) / show_range / 2.0 * canva_size
        head_canvas = head_canvas.astype(np.int32)

        for rid in sort_ids:
            score = scores[rid]
            if score < vis_thred and pred_flag[rid]:
                continue
            score = min(score * 2.0, 1.0) if pred_flag[rid] else 1.0
            color = color_map[int(pred_flag[rid])]
            for index in draw_boxes_indexes_bev:
                cv2.line(
                    canvas,
                    bottom_corners_bev[rid, index[0]],
                    bottom_corners_bev[rid, index[1]],
                    [color[0] * score, color[1] * score, color[2] * score],
                    thickness=1)
            cv2.line(
                canvas,
                center_canvas[rid],
                head_canvas[rid],
                [color[0] * score, color[1] * score, color[2] * score],
                1,
                lineType=8)

        # fuse image-view and bev
        img = np.zeros((900 * 2 + canva_size * scale_factor, 1600 * 3, 3),
                       dtype=np.uint8)
        img[:900, :, :] = np.concatenate(imgs[:3], axis=1)
        img_back = np.concatenate(
            [imgs[3][:, ::-1, :], imgs[4][:, ::-1, :], imgs[5][:, ::-1, :]],
            axis=1)
        img[900 + canva_size * scale_factor:, :, :] = img_back
        img = cv2.resize(img, (int(1600 / scale_factor * 3),
                               int(900 / scale_factor * 2 + canva_size)))
        w_begin = int((1600 * 3 / scale_factor - canva_size) // 2)
        img[int(900 / scale_factor):int(900 / scale_factor) + canva_size,
            w_begin:w_begin + canva_size, :] = canvas

        if save_format == 'image':
            img_list += [img]
        elif save_format == 'video':
            vout.write(img)
    if save_format == 'video':
        vout.release()
    return img_list
