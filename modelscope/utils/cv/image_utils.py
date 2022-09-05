import cv2
import numpy as np

from modelscope.outputs import OutputKeys
from modelscope.preprocessors.image import load_image


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
    cv2.rectangle(image, (int(box[0][0]), int(box[0][1])),
                  (int(box[1][0]), int(box[1][1])), (0, 0, 255), 2)


def realtime_object_detection_bbox_vis(image, bboxes):
    for bbox in bboxes:
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      (255, 0, 0), 2)
    return image


def draw_keypoints(output, original_image):
    poses = np.array(output[OutputKeys.POSES])
    scores = np.array(output[OutputKeys.SCORES])
    boxes = np.array(output[OutputKeys.BOXES])
    assert len(poses) == len(scores) and len(poses) == len(boxes)
    image = cv2.imread(original_image, -1)
    for i in range(len(poses)):
        draw_box(image, np.array(boxes[i]))
        draw_joints(image, np.array(poses[i]), np.array(scores[i]))
    return image


def draw_facial_expression_result(img_path, facial_expression_result):
    label_idx = facial_expression_result[OutputKeys.LABELS]
    map_list = [
        'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'
    ]
    label = map_list[label_idx]

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
