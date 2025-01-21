# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

PI = float(np.pi)


def coorx2u(x, w=1024):
    return ((x + 0.5) / w - 0.5) * 2 * np.pi


def coory2v(y, h=512):
    return ((y + 0.5) / h - 0.5) * np.pi


def u2coorx(u, w=1024):
    return (u / (2 * np.pi) + 0.5) * w - 0.5


def v2coory(v, h=512):
    return (v / np.pi + 0.5) * h - 0.5


def uv2xy(u, v, z=-50):
    c = z / np.tan(v)
    x = c * np.cos(u)
    y = c * np.sin(u)
    return x, y


def pano_connect_points(p1, p2, z=-50, w=1024, h=512):
    if p1[0] == p2[0]:
        return np.array([p1, p2], np.float32)

    u1 = coorx2u(p1[0], w)
    v1 = coory2v(p1[1], h)
    u2 = coorx2u(p2[0], w)
    v2 = coory2v(p2[1], h)

    x1, y1 = uv2xy(u1, v1, z)
    x2, y2 = uv2xy(u2, v2, z)

    if abs(p1[0] - p2[0]) < w / 2:
        pstart = np.ceil(min(p1[0], p2[0]))
        pend = np.floor(max(p1[0], p2[0]))
    else:
        pstart = np.ceil(max(p1[0], p2[0]))
        pend = np.floor(min(p1[0], p2[0]) + w)
    coorxs = (np.arange(pstart, pend + 1) % w).astype(np.float64)
    vx = x2 - x1
    vy = y2 - y1
    us = coorx2u(coorxs, w)
    ps = (np.tan(us) * x1 - y1) / (vy - np.tan(us) * vx)
    cs = np.sqrt((x1 + ps * vx)**2 + (y1 + ps * vy)**2)
    vs = np.arctan2(z, cs)
    coorys = v2coory(vs, h)

    return np.stack([coorxs, coorys], axis=-1)


def sort_xy_filter_unique(xs, ys, y_small_first=True):
    xs, ys = np.array(xs), np.array(ys)
    idx_sort = np.argsort(xs + ys / ys.max() * (int(y_small_first) * 2 - 1))
    xs, ys = xs[idx_sort], ys[idx_sort]
    _, idx_unique = np.unique(xs, return_index=True)
    xs, ys = xs[idx_unique], ys[idx_unique]
    assert np.all(np.diff(xs) > 0)
    return xs, ys


def cor_2_1d(cor, H, W):
    bon_ceil_x, bon_ceil_y = [], []
    bon_floor_x, bon_floor_y = [], []
    n_cor = len(cor)
    for i in range(n_cor // 2):
        xys = pano_connect_points(
            cor[i * 2], cor[(i * 2 + 2) % n_cor], z=-50, w=W, h=H)
        bon_ceil_x.extend(xys[:, 0])
        bon_ceil_y.extend(xys[:, 1])
    for i in range(n_cor // 2):
        xys = pano_connect_points(
            cor[i * 2 + 1], cor[(i * 2 + 3) % n_cor], z=50, w=W, h=H)
        bon_floor_x.extend(xys[:, 0])
        bon_floor_y.extend(xys[:, 1])
    bon_ceil_x, bon_ceil_y = sort_xy_filter_unique(
        bon_ceil_x, bon_ceil_y, y_small_first=True)
    bon_floor_x, bon_floor_y = sort_xy_filter_unique(
        bon_floor_x, bon_floor_y, y_small_first=False)
    bon = np.zeros((2, W))
    bon[0] = np.interp(np.arange(W), bon_ceil_x, bon_ceil_y, period=W)
    bon[1] = np.interp(np.arange(W), bon_floor_x, bon_floor_y, period=W)
    bon = ((bon + 0.5) / H - 0.5) * np.pi
    return bon


def fuv2img(fuv, coorW=1024, floorW=1024, floorH=512):
    floor_plane_x, floor_plane_y = np.meshgrid(range(floorW), range(floorH))
    floor_plane_x, floor_plane_y = -(floor_plane_y
                                     - floorH / 2), floor_plane_x - floorW / 2
    floor_plane_coridx = \
        (np.arctan2(floor_plane_y, floor_plane_x) / (2 * PI) + 0.5) * coorW - 0.5
    floor_plane = map_coordinates(
        fuv, floor_plane_coridx.reshape(1, -1), order=1, mode='wrap')
    floor_plane = floor_plane.reshape(floorH, floorW)
    return floor_plane


def np_coorx2u(coorx, coorW=1024):
    return ((coorx + 0.5) / coorW - 0.5) * 2 * PI


def np_coory2v(coory, coorH=512):
    return -((coory + 0.5) / coorH - 0.5) * PI


def np_coor2xy(coor, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512):
    coor = np.array(coor)
    u = np_coorx2u(coor[:, 0], coorW)
    v = np_coory2v(coor[:, 1], coorH)
    c = z / np.tan(v)
    x = c * np.sin(u) + floorW / 2 - 0.5
    y = -c * np.cos(u) + floorH / 2 - 0.5
    return np.hstack([x[:, None], y[:, None]])


def np_x_u_solve_y(x, u, floorW=1024, floorH=512):
    c = (x - floorW / 2 + 0.5) / np.sin(u)
    return -c * np.cos(u) + floorH / 2 - 0.5


def np_y_u_solve_x(y, u, floorW=1024, floorH=512):
    c = -(y - floorH / 2 + 0.5) / np.cos(u)
    return c * np.sin(u) + floorW / 2 - 0.5


def np_xy2coor(xy, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512):
    x = xy[:, 0] - floorW / 2 + 0.5
    y = xy[:, 1] - floorH / 2 + 0.5

    u = np.arctan2(x, -y)
    v = np.arctan(z / np.sqrt(x**2 + y**2))

    coorx = (u / (2 * PI) + 0.5) * coorW - 0.5
    coory = (-v / PI + 0.5) * coorH - 0.5

    return np.hstack([coorx[:, None], coory[:, None]])


def mean_percentile(vec, p1=25, p2=75):
    vmin = np.percentile(vec, p1)
    vmax = np.percentile(vec, p2)
    return vec[(vmin <= vec) & (vec <= vmax)].mean()


def vote(vec, tol):
    vec = np.sort(vec)
    n = np.arange(len(vec))[::-1]
    n = n[:, None] - n[None, :] + 1.0
    la = squareform(pdist(vec[:, None], 'minkowski', p=1) + 1e-9)

    invalid = (n < len(vec) * 0.4) | (la > tol)
    if (~invalid).sum() == 0 or len(vec) < tol:
        best_fit = np.median(vec)
        p_score = 0
    else:
        la[invalid] = 1e5
        n[invalid] = -1
        score = n
        max_idx = score.argmax()
        max_row = max_idx // len(vec)
        max_col = max_idx % len(vec)
        assert max_col > max_row
        best_fit = np.median(vec)
        p_score = (max_col - max_row + 1) / len(vec)

    l1_score = np.abs(vec - best_fit).mean()

    return best_fit, p_score, l1_score


def get_z1(coory0, coory1, z0=50, coorH=512):
    v0 = np_coory2v(coory0, coorH)
    v1 = np_coory2v(coory1, coorH)
    c0 = z0 / np.tan(v0)
    z1 = c0 * np.tan(v1)
    return z1


def np_refine_by_fix_z(coory0, coory1, z0=50, coorH=512):
    v0 = np_coory2v(coory0, coorH)
    v1 = np_coory2v(coory1, coorH)

    c0 = z0 / np.tan(v0)
    z1 = c0 * np.tan(v1)
    z1_mean = mean_percentile(z1)
    v1_refine = np.arctan2(z1_mean, c0)
    coory1_refine = (-v1_refine / PI + 0.5) * coorH - 0.5

    return coory1_refine, z1_mean


def infer_coory(coory0, h, z0=50, coorH=512):
    v0 = np_coory2v(coory0, coorH)
    c0 = z0 / np.tan(v0)
    z1 = z0 + h
    v1 = np.arctan2(z1, c0)
    return (-v1 / PI + 0.5) * coorH - 0.5


def get_gpid(coorx, coorW):
    gpid = np.zeros(coorW)
    gpid[np.round(coorx).astype(int)] = 1
    gpid = np.cumsum(gpid).astype(int)
    gpid[gpid == gpid[-1]] = 0
    return gpid


def get_gpid_idx(gpid, j):
    idx = np.where(gpid == j)[0]
    if idx[0] == 0 and idx[-1] != len(idx) - 1:
        _shift = -np.where(idx != np.arange(len(idx)))[0][0]
        idx = np.roll(idx, _shift)
    return idx


def gpid_two_split(xy, tpid_a, tpid_b):
    m = np.arange(len(xy)) + 1
    cum_a = np.cumsum(xy[:, tpid_a])
    cum_b = np.cumsum(xy[::-1, tpid_b])
    l1_a = cum_a / m - cum_a / (m * m)
    l1_b = cum_b / m - cum_b / (m * m)
    l1_b = l1_b[::-1]

    score = l1_a[:-1] + l1_b[1:]
    best_split = score.argmax() + 1

    va = xy[:best_split, tpid_a].mean()
    vb = xy[best_split:, tpid_b].mean()

    return va, vb


def _get_rot_rad(px, py):
    if px < 0:
        px, py = -px, -py
    rad = np.arctan2(py, px) * 180 / np.pi
    if rad > 45:
        return 90 - rad
    if rad < -45:
        return -90 - rad
    return -rad


def get_rot_rad(init_coorx,
                coory,
                z=50,
                coorW=1024,
                coorH=512,
                floorW=1024,
                floorH=512,
                tol=5):
    gpid = get_gpid(init_coorx, coorW)
    coor = np.hstack([np.arange(coorW)[:, None], coory[:, None]])
    xy = np_coor2xy(coor, z, coorW, coorH, floorW, floorH)

    rot_rad_suggestions = []
    for j in range(len(init_coorx)):
        pca = PCA(n_components=1)
        pca.fit(xy[gpid == j])
        rot_rad_suggestions.append(_get_rot_rad(*pca.components_[0]))
    rot_rad_suggestions = np.sort(rot_rad_suggestions + [1e9])

    rot_rad = np.mean(rot_rad_suggestions[:-1])
    best_rot_rad_sz = -1
    last_j = 0
    for j in range(1, len(rot_rad_suggestions)):
        if rot_rad_suggestions[j] - rot_rad_suggestions[j - 1] > tol:
            last_j = j
        elif j - last_j > best_rot_rad_sz:
            rot_rad = rot_rad_suggestions[last_j:j + 1].mean()
            best_rot_rad_sz = j - last_j

    dx = int(round(rot_rad * 1024 / 360))
    return dx, rot_rad


def gen_ww_cuboid(xy, gpid):
    assert len(np.unique(gpid)) == 4
    xy_cor = [
        {
            'type': 1,
            'val': np.median(xy[gpid == 0, 1])
        },
        {
            'type': 0,
            'val': np.median(xy[gpid == 1, 0])
        },
        {
            'type': 1,
            'val': np.median(xy[gpid == 2, 1])
        },
        {
            'type': 0,
            'val': np.median(xy[gpid == 3, 0])
        },
    ]
    return xy_cor


def gen_ww_general(init_coorx, xy, gpid, tol):
    xy_cor = []
    assert len(init_coorx) == len(np.unique(gpid))

    for j in range(len(init_coorx)):
        now_x = xy[gpid == j, 0]
        now_y = xy[gpid == j, 1]
        new_x, x_score, x_l1 = vote(now_x, tol)
        new_y, y_score, y_l1 = vote(now_y, tol)
        u0 = np_coorx2u(init_coorx[(j - 1 + len(init_coorx))
                                   % len(init_coorx)])
        u1 = np_coorx2u(init_coorx[j])
        if (x_score, -x_l1) > (y_score, -y_l1):
            xy_cor.append({
                'type': 0,
                'val': new_x,
                'score': x_score,
                'action': 'ori',
                'gpid': j,
                'u0': u0,
                'u1': u1,
                'tbd': True
            })
        else:
            xy_cor.append({
                'type': 1,
                'val': new_y,
                'score': y_score,
                'action': 'ori',
                'gpid': j,
                'u0': u0,
                'u1': u1,
                'tbd': True
            })

    while True:
        tbd = -1
        for i in range(len(xy_cor)):
            if xy_cor[i]['tbd'] and (
                    tbd == -1 or xy_cor[i]['score'] > xy_cor[tbd]['score']):
                tbd = i
        if tbd == -1:
            break

        xy_cor[tbd]['tbd'] = False
        p_idx = (tbd - 1 + len(xy_cor)) % len(xy_cor)
        n_idx = (tbd + 1) % len(xy_cor)

        num_tbd_neighbor = xy_cor[p_idx]['tbd'] + xy_cor[n_idx]['tbd']

        if num_tbd_neighbor == 2:
            continue

        if num_tbd_neighbor == 1:
            if (not xy_cor[p_idx]['tbd'] and xy_cor[p_idx]['type'] == xy_cor[tbd]['type']) or\
                    (not xy_cor[n_idx]['tbd'] and xy_cor[n_idx]['type'] == xy_cor[tbd]['type']):
                if xy_cor[tbd]['score'] >= -1:
                    xy_cor[tbd]['tbd'] = True
                    xy_cor[tbd]['score'] -= 100
                else:
                    if not xy_cor[p_idx]['tbd']:
                        insert_at = tbd
                        if xy_cor[p_idx]['type'] == 0:
                            new_val = np_x_u_solve_y(xy_cor[p_idx]['val'],
                                                     xy_cor[p_idx]['u1'])
                            new_type = 1
                        else:
                            new_val = np_y_u_solve_x(xy_cor[p_idx]['val'],
                                                     xy_cor[p_idx]['u1'])
                            new_type = 0
                    else:
                        insert_at = n_idx
                        if xy_cor[n_idx]['type'] == 0:
                            new_val = np_x_u_solve_y(xy_cor[n_idx]['val'],
                                                     xy_cor[n_idx]['u0'])
                            new_type = 1
                        else:
                            new_val = np_y_u_solve_x(xy_cor[n_idx]['val'],
                                                     xy_cor[n_idx]['u0'])
                            new_type = 0
                    new_add = {
                        'type': new_type,
                        'val': new_val,
                        'score': 0,
                        'action': 'forced infer',
                        'gpid': -1,
                        'u0': -1,
                        'u1': -1,
                        'tbd': False
                    }
                    xy_cor.insert(insert_at, new_add)
            continue

        if xy_cor[p_idx]['type'] == xy_cor[n_idx]['type']:
            if xy_cor[tbd]['type'] == xy_cor[p_idx]['type']:
                xy_cor[tbd]['type'] = (xy_cor[tbd]['type'] + 1) % 2
                xy_cor[tbd]['action'] = 'forced change'
                xy_cor[tbd]['val'] = xy[gpid == xy_cor[tbd]['gpid'],
                                        xy_cor[tbd]['type']].mean()
        else:
            tp0 = xy_cor[n_idx]['type']
            tp1 = xy_cor[p_idx]['type']
            if xy_cor[p_idx]['type'] == 0:
                val0 = np_x_u_solve_y(xy_cor[p_idx]['val'],
                                      xy_cor[p_idx]['u1'])
                val1 = np_y_u_solve_x(xy_cor[n_idx]['val'],
                                      xy_cor[n_idx]['u0'])
            else:
                val0 = np_y_u_solve_x(xy_cor[p_idx]['val'],
                                      xy_cor[p_idx]['u1'])
                val1 = np_x_u_solve_y(xy_cor[n_idx]['val'],
                                      xy_cor[n_idx]['u0'])
            new_add = [
                {
                    'type': tp0,
                    'val': val0,
                    'score': 0,
                    'action': 'forced infer',
                    'gpid': -1,
                    'u0': -1,
                    'u1': -1,
                    'tbd': False
                },
                {
                    'type': tp1,
                    'val': val1,
                    'score': 0,
                    'action': 'forced infer',
                    'gpid': -1,
                    'u0': -1,
                    'u1': -1,
                    'tbd': False
                },
            ]
            xy_cor = xy_cor[:tbd] + new_add + xy_cor[tbd + 1:]

    return xy_cor


def gen_ww(init_coorx,
           coory,
           z=50,
           coorW=1024,
           coorH=512,
           floorW=1024,
           floorH=512,
           tol=3,
           force_cuboid=True):
    gpid = get_gpid(init_coorx, coorW)
    coor = np.hstack([np.arange(coorW)[:, None], coory[:, None]])
    xy = np_coor2xy(coor, z, coorW, coorH, floorW, floorH)

    if force_cuboid:
        xy_cor = gen_ww_cuboid(xy, gpid)
    else:
        xy_cor = gen_ww_general(init_coorx, xy, gpid, tol)

    cor = []
    for j in range(len(xy_cor)):
        next_j = (j + 1) % len(xy_cor)
        if xy_cor[j]['type'] == 1:
            cor.append((xy_cor[next_j]['val'], xy_cor[j]['val']))
        else:
            cor.append((xy_cor[j]['val'], xy_cor[next_j]['val']))
    cor = np_xy2coor(np.array(cor), z, coorW, coorH, floorW, floorH)
    cor = np.roll(cor, -2 * cor[::2, 0].argmin(), axis=0)

    return cor, xy_cor
