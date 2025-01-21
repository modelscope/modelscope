# The implementation is adopted from FairMOT,
# made publicly available under the MIT License at https://github.com/ifzhang/FairMOT
import lap
import numpy as np
from scipy.spatial.distance import cdist

from modelscope.models.cv.video_multi_object_tracking.utils import \
    kalman_filter
from modelscope.models.cv.video_multi_object_tracking.utils.utils import \
    bbox_iou


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2),
                        dtype=int), tuple(range(cost_matrix.shape[0])), tuple(
                            range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious.size == 0:
        return ious

    ious = bbox_iou(atlbrs, btlbrs, True).numpy()

    return ious


def iou_distance(atracks, btracks):
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    Args:
        tracks: list[STrack]
        detections: list[BaseTrack]
        metric: str
    Returns:
        cost_matrix: np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections],
                              dtype=float)
    track_features = np.asarray([track.smooth_feat for track in tracks],
                                dtype=float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))
    return cost_matrix


def fuse_motion(kf,
                cost_matrix,
                tracks,
                detections,
                only_position=False,
                lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean,
            track.covariance,
            measurements,
            only_position,
            metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (
            1 - lambda_) * gating_distance
    return cost_matrix
