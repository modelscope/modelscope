# The implementation here is modified based on InsightFace_Pytorch, originally Apache License and publicly available
# at https://github.com/610265158/Peppa_Pig_Face_Engine
import numpy as np


class GroupTrack():

    def __init__(self):
        self.old_frame = None
        self.previous_landmarks_set = None
        self.with_landmark = True
        self.thres = 1
        self.alpha = 0.95
        self.iou_thres = 0.5

    def calculate(self, img, current_landmarks_set):
        if self.previous_landmarks_set is None:
            self.previous_landmarks_set = current_landmarks_set
            result = current_landmarks_set
        else:
            previous_lm_num = self.previous_landmarks_set.shape[0]
            if previous_lm_num == 0:
                self.previous_landmarks_set = current_landmarks_set
                result = current_landmarks_set
                return result
            else:
                result = []
                for i in range(current_landmarks_set.shape[0]):
                    not_in_flag = True
                    for j in range(previous_lm_num):
                        if self.iou(current_landmarks_set[i],
                                    self.previous_landmarks_set[j]
                                    ) > self.iou_thres:
                            result.append(
                                self.smooth(current_landmarks_set[i],
                                            self.previous_landmarks_set[j]))
                            not_in_flag = False
                            break
                    if not_in_flag:
                        result.append(current_landmarks_set[i])

        result = np.array(result)
        self.previous_landmarks_set = result

        return result

    def iou(self, p_set0, p_set1):
        rec1 = [
            np.min(p_set0[:, 0]),
            np.min(p_set0[:, 1]),
            np.max(p_set0[:, 0]),
            np.max(p_set0[:, 1])
        ]
        rec2 = [
            np.min(p_set1[:, 0]),
            np.min(p_set1[:, 1]),
            np.max(p_set1[:, 0]),
            np.max(p_set1[:, 1])
        ]

        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

        # computing the sum_area
        sum_area = S_rec1 + S_rec2

        # find the each edge of intersect rectangle
        x1 = max(rec1[0], rec2[0])
        y1 = max(rec1[1], rec2[1])
        x2 = min(rec1[2], rec2[2])
        y2 = min(rec1[3], rec2[3])

        # judge if there is an intersect
        intersect = max(0, x2 - x1) * max(0, y2 - y1)

        iou = intersect / (sum_area - intersect)
        return iou

    def smooth(self, now_landmarks, previous_landmarks):
        result = []
        for i in range(now_landmarks.shape[0]):
            x = now_landmarks[i][0] - previous_landmarks[i][0]
            y = now_landmarks[i][1] - previous_landmarks[i][1]
            dis = np.sqrt(np.square(x) + np.square(y))
            if dis < self.thres:
                result.append(previous_landmarks[i])
            else:
                result.append(
                    self.do_moving_average(now_landmarks[i],
                                           previous_landmarks[i]))

        return np.array(result)

    def do_moving_average(self, p_now, p_previous):
        p = self.alpha * p_now + (1 - self.alpha) * p_previous
        return p
