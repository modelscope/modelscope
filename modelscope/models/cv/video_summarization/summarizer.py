# Part of the implementation is borrowed and modified from PGL-SUM,
# publicly available at https://github.com/e-apostolidis/PGL-SUM

import os.path as osp
from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.video_summarization.kts.cpd_auto import cpd_auto
from modelscope.models.cv.video_summarization.pgl_sum import PGL_SUM
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


def get_change_points(video_feat, n_frame):
    video_feat = np.array(video_feat, np.float32)
    K = np.dot(video_feat, video_feat.T)
    change_points, _ = cpd_auto(
        K, ncp=min(K.shape[0] - 1, 120), vmax=2.2 / 4.0, lmin=1)
    change_points = change_points * 15
    change_points = np.concatenate(([0], change_points, [n_frame - 1]))

    temp_change_points = []
    for idx in range(len(change_points) - 1):
        segment = [change_points[idx], change_points[idx + 1] - 1]
        if idx == len(change_points) - 2:
            segment = [change_points[idx], change_points[idx + 1]]

        temp_change_points.append(segment)
    change_points = np.array(list(temp_change_points))

    temp_n_frame_per_seg = []
    for change_points_idx in range(len(change_points)):
        n_frame = change_points[change_points_idx][1] - change_points[
            change_points_idx][0]
        temp_n_frame_per_seg.append(n_frame)
    n_frame_per_seg = np.array(list(temp_n_frame_per_seg))

    return change_points, n_frame_per_seg


def knap_sack(W, wt, val, n):
    """ Maximize the value that a knapsack of capacity W can hold. You can either put the item or discard it, there is
    no concept of putting some part of item in the knapsack.

    :param int W: Maximum capacity -in frames- of the knapsack.
    :param list[int] wt: The weights (lengths -in frames-) of each video shot.
    :param list[float] val: The values (importance scores) of each video shot.
    :param int n: The number of the shots.
    :return: A list containing the indices of the selected shots.
    """
    K = [[0 for _ in range(W + 1)] for _ in range(n + 1)]

    # Build table K[][] in bottom up manner
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]],
                              K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

    selected = []
    w = W
    for i in range(n, 0, -1):
        if K[i][w] != K[i - 1][w]:
            selected.insert(0, i - 1)
            w -= wt[i - 1]

    return selected


def generate_summary(all_shot_bound, all_scores, all_nframes, all_positions):
    """ Generate the automatic machine summary, based on the video shots; the frame importance scores; the number of
    frames in the original video and the position of the sub-sampled frames of the original video.

    :param list[np.ndarray] all_shot_bound: The video shots for all the -original- testing videos.
    :param list[np.ndarray] all_scores: The calculated frame importance scores for all the sub-sampled testing videos.
    :param list[np.ndarray] all_nframes: The number of frames for all the -original- testing videos.
    :param list[np.ndarray] all_positions: The position of the sub-sampled frames for all the -original- testing videos.
    :return: A list containing the indices of the selected frames for all the -original- testing videos.
    """
    all_summaries = []
    for video_index in range(len(all_scores)):
        # Get shots' boundaries
        shot_bound = all_shot_bound[video_index]  # [number_of_shots, 2]
        frame_init_scores = all_scores[video_index]
        n_frames = all_nframes[video_index]
        positions = all_positions[video_index]

        # Compute the importance scores for the initial frame sequence (not the sub-sampled one)
        frame_scores = np.zeros(n_frames, dtype=np.float32)
        if positions.dtype != int:
            positions = positions.astype(np.int32)
        if positions[-1] != n_frames:
            positions = np.concatenate([positions, [n_frames]])
        for i in range(len(positions) - 1):
            pos_left, pos_right = positions[i], positions[i + 1]
            if i == len(frame_init_scores):
                frame_scores[pos_left:pos_right] = 0
            else:
                frame_scores[pos_left:pos_right] = frame_init_scores[i]

        # Compute shot-level importance scores by taking the average importance scores of all frames in the shot
        shot_imp_scores = []
        shot_lengths = []
        for shot in shot_bound:
            shot_lengths.append(shot[1] - shot[0] + 1)
            shot_imp_scores.append(
                (frame_scores[shot[0]:shot[1] + 1].mean()).item())

        # Select the best shots using the knapsack implementation
        final_shot = shot_bound[-1]
        final_max_length = int((final_shot[1] + 1) * 0.15)

        selected = knap_sack(final_max_length, shot_lengths, shot_imp_scores,
                             len(shot_lengths))

        # Select all frames from each selected shot (by setting their value in the summary vector to 1)
        summary = np.zeros(final_shot[1] + 1, dtype=np.int8)
        for shot in selected:
            summary[shot_bound[shot][0]:shot_bound[shot][1] + 1] = 1

        all_summaries.append(summary)

    return all_summaries


def transform_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    time = '%02d:%02d:%06.3f' % (h, m, s)
    return time


def summary_format(summary, fps):
    frames_list = []
    start_frame = -1
    end_frame = -1
    is_summary_frame = False
    for i, idx in enumerate(summary):
        if idx:
            if is_summary_frame is False:
                start_frame = i
                is_summary_frame = True
        else:
            if is_summary_frame:
                end_frame = i - 1
                frames_list.append([start_frame, end_frame])
                is_summary_frame = False

    if is_summary_frame and summary[-1] == 1:
        end_frame = len(summary) - 1
        frames_list.append([start_frame, end_frame])

    output = []
    for seg in frames_list:
        output.append({
            'frame':
            seg,
            'timestamps': [
                transform_time(seg[0] / float(fps)),
                transform_time(seg[1] / float(fps))
            ]
        })
    return output


@MODELS.register_module(
    Tasks.video_summarization, module_name=Models.video_summarization)
class PGLVideoSummarization(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the video summarization model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)

        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)

        self.loss = nn.MSELoss()
        self.model = PGL_SUM(
            input_size=1024,
            output_size=1024,
            num_segments=4,
            heads=8,
            fusion='add',
            pos_enc='absolute')
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        self.model = self.model.to(self._device)

        self.model = self._load_pretrained(self.model, model_path)

        if self.training:
            self.model.train()
        else:
            self.model.eval()

    def _train_forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        frame_features = input['frame_features']
        gtscore = input['gtscore']
        preds, attn_weights = self.model(frame_features)
        return {'loss': self.loss(preds, gtscore)}

    def _inference_forward(self, input: Dict[str,
                                             Tensor]) -> Dict[str, Tensor]:
        frame_features = input['frame_features']
        y, attn_weights = self.model(frame_features)
        return {'scores': y}

    def forward(self, input: Dict[str,
                                  Tensor]) -> Dict[str, Union[list, Tensor]]:
        """return the result by the model

        Args:
            input (Dict[str, Tensor]): the preprocessed data

        Returns:
            Dict[str, Union[list, Tensor]]: results
        """
        for key, value in input.items():
            input[key] = input[key].to(self._device)
        if self.training:
            return self._train_forward(input)
        else:
            return self._inference_forward(input)
