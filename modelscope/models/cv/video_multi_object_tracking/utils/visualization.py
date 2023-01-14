# The implementation is adopted from FairMOT,
# made publicly available under the MIT License at https://github.com/ifzhang/FairMOT
import cv2
import numpy as np


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(image,
                  tlwhs,
                  obj_ids,
                  scores=None,
                  frame_id=0,
                  fps=0.,
                  ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 500.))

    cv2.putText(
        im,
        'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
        (0, int(15 * text_scale)),
        cv2.FONT_HERSHEY_PLAIN,
        text_scale, (0, 0, 255),
        thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(
            im,
            intbox[0:2],
            intbox[2:4],
            color=color,
            thickness=line_thickness)
        cv2.putText(
            im,
            id_text, (intbox[0], intbox[1] + 30),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale, (0, 0, 255),
            thickness=text_thickness)
    return im


def show_multi_object_tracking_result(video_in_path, bboxes, video_save_path):
    cap = cv2.VideoCapture(video_in_path)
    frame_idx = 0
    while (cap.isOpened()):
        frame_idx += 1
        success, frame = cap.read()
        if not success:
            if frame_idx == 1:
                raise Exception(video_in_path,
                                ' can not be correctly decoded by OpenCV.')
            else:
                break
        cur_frame_boxes = []
        cur_obj_ids = []
        for box in bboxes:
            if box[0] == frame_idx:
                cur_frame_boxes.append(
                    [box[2], box[3], box[4] - box[2], box[5] - box[3]])
                cur_obj_ids.append(box[1])
        if frame_idx == 1:
            size = (frame.shape[1], frame.shape[0])
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            video_writer = cv2.VideoWriter(video_save_path, fourcc,
                                           cap.get(cv2.CAP_PROP_FPS), size,
                                           True)
        frame = plot_tracking(frame, cur_frame_boxes, cur_obj_ids, frame_idx)
        video_writer.write(frame)
    video_writer.release
    cap.release()
