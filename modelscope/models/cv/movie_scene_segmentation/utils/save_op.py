# The implementation here is modified based on SceneSeg,
# originally Apache 2.0 License and publicly avaialbe at https://github.com/AnyiRao/SceneSeg
import os
import os.path as osp
import subprocess

import cv2
import numpy as np
from tqdm import tqdm


def get_pred_boundary(pred_dict, threshold=0.5):
    pred = pred_dict['pred']
    tmp = (pred > threshold).astype(np.int32)
    anno_dict = {}
    for idx in range(len(tmp)):
        anno_dict.update({str(pred_dict['sid'][idx]).zfill(4): int(tmp[idx])})
    return anno_dict


def pred2scene(shot2keyf, anno_dict):
    scene_list, pair_list = get_demo_scene_list(shot2keyf, anno_dict)

    scene_dict_lst = []
    shot_num = len(shot2keyf)
    shot_dict_lst = []
    for item in shot2keyf:
        tmp = item.split(' ')
        shot_dict_lst.append({
            'frame': [tmp[0], tmp[1]],
            'timestamps': [tmp[-2], tmp[-1]]
        })
    assert len(scene_list) == len(pair_list)
    for scene_ind, scene_item in enumerate(scene_list):
        scene_dict_lst.append({
            'shot': pair_list[scene_ind],
            'frame': scene_item[0],
            'timestamps': scene_item[1]
        })

    return scene_dict_lst, scene_list, shot_num, shot_dict_lst


def scene2video(source_movie_fn, scene_list, thres):

    vcap = cv2.VideoCapture(source_movie_fn)
    fps = vcap.get(cv2.CAP_PROP_FPS)  # video.fps
    out_video_dir_fn = os.path.join(os.getcwd(),
                                    f'pred_result/scene_video_{thres}')
    os.makedirs(out_video_dir_fn, exist_ok=True)

    for scene_ind, scene_item in tqdm(enumerate(scene_list)):
        scene = str(scene_ind).zfill(4)
        start_frame = int(scene_item[0][0])
        end_frame = int(scene_item[0][1])
        start_time, end_time = start_frame / fps, end_frame / fps
        duration_time = end_time - start_time
        out_video_fn = os.path.join(out_video_dir_fn,
                                    'scene_{}.mp4'.format(scene))
        if os.path.exists(out_video_fn):
            continue
        call_list = ['ffmpeg']
        call_list += ['-v', 'quiet']
        call_list += [
            '-y', '-ss',
            str(start_time), '-t',
            str(duration_time), '-i', source_movie_fn
        ]
        call_list += ['-map_chapters', '-1']
        call_list += [out_video_fn]
        subprocess.call(call_list)
    return osp.join(os.getcwd(), 'pred_result')


def get_demo_scene_list(shot2keyf, anno_dict):
    pair_list = get_pair_list(anno_dict)

    scene_list = []
    for pair in pair_list:
        start_shot, end_shot = int(pair[0]), int(pair[-1])
        start_frame = shot2keyf[start_shot].split(' ')[0]
        end_frame = shot2keyf[end_shot].split(' ')[1]
        start_timestamp = shot2keyf[start_shot].split(' ')[-2]
        end_timestamp = shot2keyf[end_shot].split(' ')[-1]
        scene_list.append([[start_frame, end_frame],
                           [start_timestamp, end_timestamp]])
    return scene_list, pair_list


def get_pair_list(anno_dict):
    sort_anno_dict_key = sorted(anno_dict.keys())
    tmp = 0
    tmp_list = []
    tmp_label_list = []
    anno_list = []
    anno_label_list = []
    for key in sort_anno_dict_key:
        value = anno_dict.get(key)
        tmp += value
        tmp_list.append(key)
        tmp_label_list.append(value)
        if tmp == 1:
            anno_list.append(tmp_list)
            anno_label_list.append(tmp_label_list)
            tmp = 0
            tmp_list = []
            tmp_label_list = []
            continue
        if key == sort_anno_dict_key[-1]:
            if len(tmp_list) > 0:
                anno_list.append(tmp_list)
                anno_label_list.append(tmp_label_list)
    if len(anno_list) == 0:
        return None
    while [] in anno_list:
        anno_list.remove([])
    tmp_anno_list = [anno_list[0]]
    pair_list = []
    for ind in range(len(anno_list) - 1):
        cont_count = int(anno_list[ind + 1][0]) - int(anno_list[ind][-1])
        if cont_count > 1:
            pair_list.extend(tmp_anno_list)
            tmp_anno_list = [anno_list[ind + 1]]
            continue
        tmp_anno_list.append(anno_list[ind + 1])
    pair_list.extend(tmp_anno_list)
    return pair_list
