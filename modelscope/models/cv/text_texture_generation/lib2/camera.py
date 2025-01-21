# customized
import sys

import numpy as np
import torch
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform
from sklearn.metrics.pairwise import cosine_similarity

from modelscope.models.cv.text_texture_generation.lib2.init_view import \
    VIEWPOINTS

sys.path.append('.')

# ---------------- UTILS ----------------------


def degree_to_radian(d):
    return d * np.pi / 180


def radian_to_degree(r):
    return 180 * r / np.pi


def xyz_to_polar(xyz):
    """ assume y-axis is the up axis """

    x, y, z = xyz

    theta = 180 * np.arccos(z) / np.pi
    phi = 180 * np.arccos(y) / np.pi

    return theta, phi


def polar_to_xyz(theta, phi, dist):
    """ assume y-axis is the up axis """

    theta = degree_to_radian(theta)
    phi = degree_to_radian(phi)

    x = np.sin(phi) * np.sin(theta) * dist
    y = np.cos(phi) * dist
    z = np.sin(phi) * np.cos(theta) * dist

    return [x, y, z]


# ---------------- VIEWPOINTS ----------------------


def filter_viewpoints(pre_viewpoints: dict, viewpoints: dict):
    """ return the binary mask of viewpoints to be filtered """

    filter_mask = [0 for _ in viewpoints.keys()]
    for i, v in viewpoints.items():
        x_v, y_v, z_v = polar_to_xyz(v['azim'], 90 - v['elev'], v['dist'])

        for _, pv in pre_viewpoints.items():
            x_pv, y_pv, z_pv = polar_to_xyz(pv['azim'], 90 - pv['elev'],
                                            pv['dist'])
            sim = cosine_similarity(
                np.array([[x_v, y_v, z_v]]), np.array([[x_pv, y_pv, z_pv]]))[0,
                                                                             0]

            if sim > 0.9:
                filter_mask[i] = 1

    return filter_mask


def init_viewpoints(init_dist,
                    init_elev,
                    init_azim,
                    use_principle=True,
                    use_shapenet=False,
                    use_objaverse=False):
    sample_space = 12
    (dist_list, elev_list, azim_list,
     sector_list) = init_predefined_viewpoints(sample_space, init_dist,
                                               init_elev)

    # punishments for views -> in case always selecting the same view
    view_punishments = [1 for _ in range(len(dist_list))]

    if use_principle:
        (dist_list, elev_list, azim_list, sector_list,
         view_punishments) = init_principle_viewpoints(dist_list, elev_list,
                                                       azim_list, sector_list,
                                                       view_punishments,
                                                       use_shapenet,
                                                       use_objaverse)
    azim_list = [v - init_azim for v in azim_list]
    elev_list = [v - init_elev for v in elev_list]

    return dist_list, elev_list, azim_list, sector_list, view_punishments


def init_principle_viewpoints(dist_list,
                              elev_list,
                              azim_list,
                              sector_list,
                              view_punishments,
                              use_shapenet=False,
                              use_objaverse=False):
    if use_shapenet:
        key = 'shapenet'

        pre_elev_list = [v for v in VIEWPOINTS[key]['elev']]
        pre_azim_list = [v for v in VIEWPOINTS[key]['azim']]
        pre_sector_list = [v for v in VIEWPOINTS[key]['sector']]

        num_principle = 10
        pre_dist_list = [dist_list[0] for _ in range(num_principle)]
        pre_view_punishments = [0 for _ in range(num_principle)]

    elif use_objaverse:
        key = 'objaverse'

        pre_elev_list = [v for v in VIEWPOINTS[key]['elev']]
        pre_azim_list = [v for v in VIEWPOINTS[key]['azim']]
        pre_sector_list = [v for v in VIEWPOINTS[key]['sector']]

        num_principle = 10
        pre_dist_list = [dist_list[0] for _ in range(num_principle)]
        pre_view_punishments = [0 for _ in range(num_principle)]
    else:
        num_principle = 12
        pre_elev_list = [v for v in VIEWPOINTS[num_principle]['elev']]
        pre_azim_list = [v for v in VIEWPOINTS[num_principle]['azim']]
        pre_sector_list = [v for v in VIEWPOINTS[num_principle]['sector']]
        pre_dist_list = [dist_list[0] for _ in range(num_principle)]
        pre_view_punishments = [0 for _ in range(num_principle)]

    dist_list = pre_dist_list + dist_list
    elev_list = pre_elev_list + elev_list
    azim_list = pre_azim_list + azim_list
    sector_list = pre_sector_list + sector_list
    view_punishments = pre_view_punishments + view_punishments

    return dist_list, elev_list, azim_list, sector_list, view_punishments


def init_predefined_viewpoints(sample_space, init_dist, init_elev):
    viewpoints = VIEWPOINTS[sample_space]

    assert sample_space == len(viewpoints['sector'])

    dist_list = [init_dist
                 for _ in range(sample_space)]  # always the same dist
    elev_list = [viewpoints['elev'][i] for i in range(sample_space)]
    azim_list = [viewpoints['azim'][i] for i in range(sample_space)]
    sector_list = [viewpoints['sector'][i] for i in range(sample_space)]

    return dist_list, elev_list, azim_list, sector_list


def init_camera(dist, elev, azim, image_size, device):
    R, T = look_at_view_transform(dist, elev, azim)
    image_size = torch.tensor([image_size, image_size]).unsqueeze(0)
    T[0][2] = dist
    cameras = PerspectiveCameras(
        R=R, T=T, device=device, image_size=image_size)

    return cameras
