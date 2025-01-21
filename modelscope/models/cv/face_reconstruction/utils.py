# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import math
import os
import os.path as osp
from array import array

import cv2
import numba
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.io import loadmat, savemat


def img_value_rescale(img, old_range: list, new_range: list):
    assert len(old_range) == 2
    assert len(new_range) == 2
    img = (img - old_range[0]) / (old_range[1] - old_range[0]) * (
        new_range[1] - new_range[0]) + new_range[0]
    return img


def resize_on_long_side(img, long_side=800):
    src_height = img.shape[0]
    src_width = img.shape[1]

    if src_height > src_width:
        scale = long_side * 1.0 / src_height
        _img = cv2.resize(
            img, (int(src_width * scale), long_side),
            interpolation=cv2.INTER_CUBIC)

    else:
        scale = long_side * 1.0 / src_width
        _img = cv2.resize(
            img, (long_side, int(src_height * scale)),
            interpolation=cv2.INTER_CUBIC)

    return _img, scale


def get_mg_layer(src, gt, skin_mask=None):
    """
    src, gt shape: [h, w, 3] value: [0, 1]
    return: mg, shape: [h, w, 1] value: [0, 1]
    """
    mg = (src * src - gt + 1e-10) / (2 * src * src - 2 * src + 2e-10)
    mg[mg < 0] = 0.5
    mg[mg > 1] = 0.5

    diff_abs = np.abs(gt - src)
    mg[diff_abs < (1 / 255.0)] = 0.5

    if skin_mask is not None:
        mg[skin_mask == 0] = 0.5

    return mg


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def spread_flow(length, spread_ratio=2):
    Flow = np.zeros(shape=(length, length, 2), dtype=np.float32)
    mag = np.zeros(shape=(length, length), dtype=np.float32)

    radius = length * 0.5
    for h in range(Flow.shape[0]):
        for w in range(Flow.shape[1]):

            if (h - length // 2)**2 + (w - length // 2)**2 <= radius**2:
                Flow[h, w, 0] = -(w - length // 2)
                Flow[h, w, 1] = -(h - length // 2)

                distance = np.sqrt((w - length // 2)**2 + (h - length // 2)**2)

                if distance <= radius / 2.0:
                    mag[h, w] = 2.0 / radius * distance
                else:
                    mag[h, w] = -2.0 / radius * distance + 2.0

    _, ang = cv2.cartToPolar(Flow[..., 0] + 1e-8, Flow[..., 1] + 1e-8)

    mag *= spread_ratio

    x, y = cv2.polarToCart(mag, ang, angleInDegrees=False)
    Flow = np.dstack((x, y))

    return Flow


@numba.jit(nopython=True, parallel=True)
def bilinear_interp(x, y, v11, v12, v21, v22):
    t = 0.2

    if x < t and y < t:
        return v11
    elif x < t and y > 1 - t:
        return v12
    elif x > 1 - t and y < t:
        return v21
    elif x > 1 - t and y > 1 - t:
        return v22
    else:
        result = (v11 * (1 - y) + v12 * y) * (1 - x) + \
                 (v21 * (1 - y) + v22 * y) * x
        if result < 0:
            result = 0

        if result > 255:
            result = 255
        return result


@numba.jit(nopython=True, parallel=True)
def image_warp_grid1(rDx, rDy, oriImg, transRatio, pads):
    # assert oriImg.dtype == np.uint8
    srcW = oriImg.shape[1]
    srcH = oriImg.shape[0]

    padTop, padBottom, padLeft, padRight = pads

    left_bound = padLeft + 1
    right_bound = srcW - padRight
    bottom_bound = srcH - padBottom
    top_bound = padTop + 1

    newImg = oriImg.copy()

    for i in range(srcH):
        for j in range(srcW):
            _i = i
            _j = j

            deltaX = rDx[_i, _j]
            deltaY = rDy[_i, _j]

            if abs(deltaX) < 0.2 and abs(deltaY) < 0.2:
                continue

            nx = _j + deltaX * transRatio
            ny = _i + deltaY * transRatio

            if nx >= srcW - padRight:
                if nx > srcW - 1:
                    nx = srcW - 1

                if _j < right_bound:
                    right_bound = _j

            if ny >= srcH - padBottom:
                if ny > srcH - 1:
                    ny = srcH - 1

                if _i < bottom_bound:
                    bottom_bound = _i

            if nx < padLeft:
                if nx < 0:
                    nx = 0

                if _j + 1 > left_bound:
                    left_bound = _j + 1

            if ny < padTop:
                if ny < 0:
                    ny = 0

                if _i + 1 > top_bound:
                    top_bound = _i + 1

            nxi = int(math.floor(nx))
            nyi = int(math.floor(ny))
            nxi1 = int(math.ceil(nx))
            nyi1 = int(math.ceil(ny))

            if nxi < 0:
                nxi = 0
            if nxi > oriImg.shape[1] - 1:
                nxi = oriImg.shape[1] - 1

            if nxi1 < 0:
                nxi1 = 0
            if nxi1 > oriImg.shape[1] - 1:
                nxi1 = oriImg.shape[1] - 1

            if nyi < 0:
                nyi = 0
            if nyi > oriImg.shape[0] - 1:
                nyi = oriImg.shape[0] - 1

            if nyi1 < 0:
                nyi1 = 0
            if nyi1 > oriImg.shape[0] - 1:
                nyi1 = oriImg.shape[0] - 1

            for ll in range(3):
                newImg[_i, _j,
                       ll] = bilinear_interp(ny - nyi, nx - nxi,
                                             oriImg[nyi, nxi,
                                                    ll], oriImg[nyi, nxi1, ll],
                                             oriImg[nyi1, nxi,
                                                    ll], oriImg[nyi1, nxi1,
                                                                ll])

    return newImg, top_bound, bottom_bound, left_bound, right_bound


def warp(x, flow, mode='bilinear', padding_mode='zeros', coff=0.1):
    """

    Args:
        x: [n, c, h, w]
        flow: [n, h, w, 2]
        mode:
        padding_mode:
        coff:

    Returns:

    """
    n, c, h, w = x.size()
    yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
    xv = xv.float() / (w - 1) * 2.0 - 1
    yv = yv.float() / (h - 1) * 2.0 - 1
    '''
    grid[0,:,:,0] =
    -1, .....1
    -1, .....1
    -1, .....1

    grid[0,:,:,1] =
    -1,  -1, -1
     ;        ;
     1,   1,  1

    '''

    if torch.cuda.is_available():
        grid = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)),
                         -1).unsqueeze(0).cuda()
    else:
        grid = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), -1).unsqueeze(0)
    grid_x = grid + 2 * flow * coff
    warp_x = F.grid_sample(x, grid_x, mode=mode, padding_mode=padding_mode)
    return warp_x


# load expression basis
def LoadExpBasis(bfm_folder='asset/BFM'):
    n_vertex = 53215
    Expbin = open(osp.join(bfm_folder, 'Exp_Pca.bin'), 'rb')
    exp_dim = array('i')
    exp_dim.fromfile(Expbin, 1)
    expMU = array('f')
    expPC = array('f')
    expMU.fromfile(Expbin, 3 * n_vertex)
    expPC.fromfile(Expbin, 3 * exp_dim[0] * n_vertex)
    Expbin.close()

    expPC = np.array(expPC)
    expPC = np.reshape(expPC, [exp_dim[0], -1])
    expPC = np.transpose(expPC)

    expEV = np.loadtxt(osp.join(bfm_folder, 'std_exp.txt'))

    return expPC, expEV


# transfer original BFM09 to our face model
def transferBFM09(bfm_folder='BFM'):
    print('Transfer BFM09 to BFM_model_front......')
    original_BFM = loadmat(osp.join(bfm_folder, '01_MorphableModel.mat'))
    shapePC = original_BFM['shapePC']  # shape basis, 160470*199
    shapeEV = original_BFM['shapeEV']  # corresponding eigen value, 199*1
    shapeMU = original_BFM['shapeMU']  # mean face, 160470*1
    texPC = original_BFM['texPC']  # texture basis, 160470*199
    texEV = original_BFM['texEV']  # eigen value, 199*1
    texMU = original_BFM['texMU']  # mean texture, 160470*1

    expPC, expEV = LoadExpBasis()

    # transfer BFM09 to our face model

    idBase = shapePC * np.reshape(shapeEV, [-1, 199])
    idBase = idBase / 1e5  # unify the scale to decimeter
    idBase = idBase[:, :80]  # use only first 80 basis

    exBase = expPC * np.reshape(expEV, [-1, 79])
    exBase = exBase / 1e5  # unify the scale to decimeter
    exBase = exBase[:, :64]  # use only first 64 basis

    texBase = texPC * np.reshape(texEV, [-1, 199])
    texBase = texBase[:, :80]  # use only first 80 basis

    # our face model is cropped along face landmarks and contains only 35709 vertex.
    # original BFM09 contains 53490 vertex, and expression basis provided by Guo et al. contains 53215 vertex.
    # thus we select corresponding vertex to get our face model.

    index_exp = loadmat(osp.join(bfm_folder, 'BFM_front_idx.mat'))
    index_exp = index_exp['idx'].astype(
        np.int32) - 1  # starts from 0 (to 53215)

    index_shape = loadmat(osp.join(bfm_folder, 'BFM_exp_idx.mat'))
    index_shape = index_shape['trimIndex'].astype(
        np.int32) - 1  # starts from 0 (to 53490)
    index_shape = index_shape[index_exp]

    idBase = np.reshape(idBase, [-1, 3, 80])
    idBase = idBase[index_shape, :, :]
    idBase = np.reshape(idBase, [-1, 80])

    texBase = np.reshape(texBase, [-1, 3, 80])
    texBase = texBase[index_shape, :, :]
    texBase = np.reshape(texBase, [-1, 80])

    exBase = np.reshape(exBase, [-1, 3, 64])
    exBase = exBase[index_exp, :, :]
    exBase = np.reshape(exBase, [-1, 64])

    meanshape = np.reshape(shapeMU, [-1, 3]) / 1e5
    meanshape = meanshape[index_shape, :]
    meanshape = np.reshape(meanshape, [1, -1])

    meantex = np.reshape(texMU, [-1, 3])
    meantex = meantex[index_shape, :]
    meantex = np.reshape(meantex, [1, -1])

    # other info contains triangles, region used for computing photometric loss,
    # region used for skin texture regularization, and 68 landmarks index etc.
    other_info = loadmat(osp.join(bfm_folder, 'facemodel_info.mat'))
    frontmask2_idx = other_info['frontmask2_idx']
    skinmask = other_info['skinmask']
    keypoints = other_info['keypoints']
    point_buf = other_info['point_buf']
    tri = other_info['tri']
    tri_mask2 = other_info['tri_mask2']

    # save our face model
    savemat(
        osp.join(bfm_folder, 'BFM_model_front.mat'), {
            'meanshape': meanshape,
            'meantex': meantex,
            'idBase': idBase,
            'exBase': exBase,
            'texBase': texBase,
            'tri': tri,
            'point_buf': point_buf,
            'tri_mask2': tri_mask2,
            'keypoints': keypoints,
            'frontmask2_idx': frontmask2_idx,
            'skinmask': skinmask
        })


# load landmarks for standard face, which is used for image preprocessing
def load_lm3d(bfm_folder):

    Lm3D = loadmat(osp.join(bfm_folder, 'similarity_Lm3D_all.mat'))
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    value_list = [
        Lm3D[lm_idx[0], :],
        np.mean(Lm3D[lm_idx[[1, 2]], :], 0),
        np.mean(Lm3D[lm_idx[[3, 4]], :], 0), Lm3D[lm_idx[5], :],
        Lm3D[lm_idx[6], :]
    ]
    Lm3D = np.stack(value_list, axis=0)
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

    return Lm3D


def mesh_to_string(mesh):
    out_string = ''
    out_string += '# Create by HRN\n'

    if 'colors' in mesh:
        for i, v in enumerate(mesh['vertices']):
            out_string += \
                'v {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                    v[0], v[1], v[2], mesh['colors'][i][0],
                    mesh['colors'][i][1], mesh['colors'][i][2])
    else:
        for v in mesh['vertices']:
            out_string += 'v {:.6f} {:.6f} {:.6f}\n'.format(v[0], v[1], v[2])

    if 'UVs' in mesh:
        for uv in mesh['UVs']:
            out_string += 'vt {:.6f} {:.6f}\n'.format(uv[0], uv[1])

    if 'normals' in mesh:
        for vn in mesh['normals']:
            out_string += 'vn {:.6f} {:.6f} {:.6f}\n'.format(
                vn[0], vn[1], vn[2])

    if 'faces' in mesh:
        for ind, face in enumerate(mesh['faces']):
            if 'faces_uv' in mesh or 'faces_normal' in mesh or 'UVs' in mesh:
                if 'faces_uv' in mesh:
                    face_uv = mesh['faces_uv'][ind]
                else:
                    face_uv = face
                if 'faces_normal' in mesh:
                    face_normal = mesh['faces_normal'][ind]
                else:
                    face_normal = face
                row = 'f ' + ' '.join([
                    '{}/{}/{}'.format(face[i], face_uv[i], face_normal[i])
                    for i in range(len(face))
                ]) + '\n'
            else:
                row = 'f ' + ' '.join(
                    ['{}'.format(face[i]) for i in range(len(face))]) + '\n'
            out_string += row

    return out_string


def write_obj(save_path, mesh):
    save_dir = os.path.dirname(save_path)
    save_name = os.path.splitext(os.path.basename(save_path))[0]

    if 'texture_map' in mesh:
        cv2.imwrite(
            os.path.join(save_dir, save_name + '.jpg'), mesh['texture_map'])

        with open(os.path.join(save_dir, save_name + '.mtl'), 'w') as wf:
            wf.write('# Created by HRN\n')
            wf.write('newmtl material_0\n')
            wf.write('Ka 1.000000 0.000000 0.000000\n')
            wf.write('Kd 1.000000 1.000000 1.000000\n')
            wf.write('Ks 0.000000 0.000000 0.000000\n')
            wf.write('Tr 0.000000\n')
            wf.write('illum 0\n')
            wf.write('Ns 0.000000\n')
            wf.write('map_Kd {}\n'.format(save_name + '.jpg'))

    with open(save_path, 'w') as wf:
        if 'texture_map' in mesh:
            wf.write('# Create by HRN\n')
            wf.write('mtllib ./{}.mtl\n'.format(save_name))

        if 'colors' in mesh:
            for i, v in enumerate(mesh['vertices']):
                wf.write(
                    'v {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                        v[0], v[1], v[2], mesh['colors'][i][0],
                        mesh['colors'][i][1], mesh['colors'][i][2]))
        else:
            for v in mesh['vertices']:
                wf.write('v {:.6f} {:.6f} {:.6f}\n'.format(v[0], v[1], v[2]))

        if 'UVs' in mesh:
            for uv in mesh['UVs']:
                wf.write('vt {:.6f} {:.6f}\n'.format(uv[0], uv[1]))

        if 'normals' in mesh:
            for vn in mesh['normals']:
                wf.write('vn {:.6f} {:.6f} {:.6f}\n'.format(
                    vn[0], vn[1], vn[2]))

        if 'faces' in mesh:
            for ind, face in enumerate(mesh['faces']):
                if 'faces_uv' in mesh or 'faces_normal' in mesh or 'UVs' in mesh:
                    if 'faces_uv' in mesh:
                        face_uv = mesh['faces_uv'][ind]
                    else:
                        face_uv = face
                    if 'faces_normal' in mesh:
                        face_normal = mesh['faces_normal'][ind]
                    else:
                        face_normal = face
                    row = 'f ' + ' '.join([
                        '{}/{}/{}'.format(face[i], face_uv[i], face_normal[i])
                        for i in range(len(face))
                    ]) + '\n'
                else:
                    row = 'f ' + ' '.join(
                        ['{}'.format(face[i])
                         for i in range(len(face))]) + '\n'
                wf.write(row)


def read_obj(obj_path, print_shape=True):
    with open(obj_path, 'r') as f:
        bfm_lines = f.readlines()

    vertices = []
    faces = []
    uvs = []
    vns = []
    faces_uv = []
    faces_normal = []
    max_face_length = 0
    for line in bfm_lines:
        if line[:2] == 'v ':
            vertex = [
                float(a) for a in line.strip().split(' ')[1:] if len(a) > 0
            ]
            vertices.append(vertex)

        if line[:2] == 'f ':
            items = line.strip().split(' ')[1:]
            face = [int(a.split('/')[0]) for a in items if len(a) > 0]
            max_face_length = max(max_face_length, len(face))
            if len(faces) > 0 and len(face) != len(faces[0]):
                continue
            faces.append(face)

            if '/' in items[0] and len(items[0].split('/')[1]) > 0:
                face_uv = [int(a.split('/')[1]) for a in items if len(a) > 0]
                faces_uv.append(face_uv)

            if '/' in items[0] and len(items[0].split('/')) >= 3 and len(
                    items[0].split('/')[2]) > 0:
                face_normal = [
                    int(a.split('/')[2]) for a in items if len(a) > 0
                ]
                faces_normal.append(face_normal)

        if line[:3] == 'vt ':
            items = line.strip().split(' ')[1:]
            uv = [float(a) for a in items if len(a) > 0]
            uvs.append(uv)

        if line[:3] == 'vn ':
            items = line.strip().split(' ')[1:]
            vn = [float(a) for a in items if len(a) > 0]
            vns.append(vn)

    vertices = np.array(vertices).astype(np.float32)
    if max_face_length <= 3:
        faces = np.array(faces).astype(np.int32)

    if vertices.shape[1] == 3:
        mesh = {
            'vertices': vertices,
            'faces': faces,
        }
    else:
        mesh = {
            'vertices': vertices[:, :3],
            'colors': vertices[:, 3:],
            'faces': faces,
        }

    if len(uvs) > 0:
        uvs = np.array(uvs).astype(np.float32)
        mesh['uvs'] = uvs

    if len(vns) > 0:
        vns = np.array(vns).astype(np.float32)
        mesh['vns'] = vns

    if len(faces_uv) > 0:
        if max_face_length <= 3:
            faces_uv = np.array(faces_uv).astype(np.int32)
        mesh['faces_uv'] = faces_uv

    if len(faces_normal) > 0:
        if max_face_length <= 3:
            faces_normal = np.array(faces_normal).astype(np.int32)
        mesh['faces_normal'] = faces_normal

    return mesh


# calculating least square problem for image alignment
def POS(xp, x):
    npts = xp.shape[1]

    A = np.zeros([2 * npts, 8])

    A[0:2 * npts - 1:2, 0:3] = x.transpose()
    A[0:2 * npts - 1:2, 3] = 1

    A[1:2 * npts:2, 4:7] = x.transpose()
    A[1:2 * npts:2, 7] = 1

    b = np.reshape(xp.transpose(), [2 * npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
    t = np.stack([sTx, sTy], axis=0)

    return t, s


# bounding box for 68 landmark detection
def BBRegression(points, params):
    w1 = params['W1']
    b1 = params['B1']
    w2 = params['W2']
    b2 = params['B2']
    data = points.copy()
    data = data.reshape([5, 2])
    data_mean = np.mean(data, axis=0)
    x_mean = data_mean[0]
    y_mean = data_mean[1]
    data[:, 0] = data[:, 0] - x_mean
    data[:, 1] = data[:, 1] - y_mean

    rms = np.sqrt(np.sum(data**2) / 5)
    data = data / rms
    data = data.reshape([1, 10])
    data = np.transpose(data)
    inputs = np.matmul(w1, data) + b1
    inputs = 2 / (1 + np.exp(-2 * inputs)) - 1
    inputs = np.matmul(w2, inputs) + b2
    inputs = np.transpose(inputs)
    x = inputs[:, 0] * rms + x_mean
    y = inputs[:, 1] * rms + y_mean
    w = 224 / inputs[:, 2] * rms
    rects = [x, y, w, w]
    return np.array(rects).reshape([4])


# utils for landmark detection
def img_padding(img, box):
    success = True
    bbox = box.copy()
    res = np.zeros([2 * img.shape[0], 2 * img.shape[1], 3])
    res[img.shape[0] // 2:img.shape[0] + img.shape[0] // 2,
        img.shape[1] // 2:img.shape[1] + img.shape[1] // 2] = img

    bbox[0] = bbox[0] + img.shape[1] // 2
    bbox[1] = bbox[1] + img.shape[0] // 2
    if bbox[0] < 0 or bbox[1] < 0:
        success = False
    return res, bbox, success


# utils for landmark detection
def crop(img, bbox):
    padded_img, padded_bbox, flag = img_padding(img, bbox)
    if flag:
        crop_img = padded_img[padded_bbox[1]:padded_bbox[1] + padded_bbox[3],
                              padded_bbox[0]:padded_bbox[0] + padded_bbox[2]]
        crop_img = cv2.resize(
            crop_img.astype(np.uint8), (224, 224),
            interpolation=cv2.INTER_CUBIC)
        scale = 224 / padded_bbox[3]
        return crop_img, scale
    else:
        return padded_img, 0


# utils for landmark detection
def scale_trans(img, lm, t, s):
    imgw = img.shape[1]
    imgh = img.shape[0]
    M_s = np.array(
        [[1, 0, -t[0] + imgw // 2 + 0.5], [0, 1, -imgh // 2 + t[1]]],
        dtype=np.float32)
    img = cv2.warpAffine(img, M_s, (imgw, imgh))
    w = int(imgw / s * 100)
    h = int(imgh / s * 100)
    img = cv2.resize(img, (w, h))
    lm = np.stack([lm[:, 0] - t[0] + imgw // 2, lm[:, 1] - t[1] + imgh // 2],
                  axis=1) / s * 100

    left = w // 2 - 112
    up = h // 2 - 112
    bbox = [left, up, 224, 224]
    cropped_img, scale2 = crop(img, bbox)
    assert (scale2 != 0)
    t1 = np.array([bbox[0], bbox[1]])

    # back to raw img s * crop + s * t1 + t2
    t1 = np.array([w // 2 - 112, h // 2 - 112])
    scale = s / 100
    t2 = np.array([t[0] - imgw / 2, t[1] - imgh / 2])
    inv = (scale / scale2, scale * t1 + t2.reshape([2]))
    return cropped_img, inv


# utils for landmark detection
def align_for_lm(img, five_points, params):
    five_points = np.array(five_points).reshape([1, 10])
    bbox = BBRegression(five_points, params)
    assert (bbox[2] != 0)
    bbox = np.round(bbox).astype(np.int32)
    crop_img, scale = crop(img, bbox)
    return crop_img, scale, bbox


# resize and crop images for face reconstruction
def resize_n_crop_img(img, lm, t, s, target_size=224., mask=None):
    w0, h0 = img.size
    w = (w0 * s).astype(np.int32)
    h = (h0 * s).astype(np.int32)
    left = (w / 2 - target_size / 2 + float(
        (t[0] - w0 / 2) * s)).astype(np.int32)
    right = left + target_size
    up = (h / 2 - target_size / 2 + float(
        (h0 / 2 - t[1]) * s)).astype(np.int32)
    below = up + target_size

    new_img = img.resize((w, h), resample=Image.BICUBIC)
    new_img = new_img.crop((left, up, right, below))

    if mask is not None:
        mask = mask.resize((w, h), resample=Image.BICUBIC)
        mask = mask.crop((left, up, right, below))

    new_lm = np.stack([lm[:, 0] - t[0] + w0 / 2, lm[:, 1] - t[1] + h0 / 2],
                      axis=1) * s
    new_lm = new_lm - np.reshape(
        np.array([(w / 2 - target_size / 2),
                  (h / 2 - target_size / 2)]), [1, 2])

    return new_img, new_lm, mask


# utils for face reconstruction
def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    value_list = [
        lm[lm_idx[0], :],
        np.mean(lm[lm_idx[[1, 2]], :], 0),
        np.mean(lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]
    ]
    lm5p = np.stack(value_list, axis=0)
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p


# utils for face reconstruction
def align_img(img, lm, lm3D, mask=None, target_size=224., rescale_factor=102.):
    """
    Return:
        transparams        --numpy.array  (raw_W, raw_H, scale, tx, ty)
        img_new            --PIL.Image  (target_size, target_size, 3)
        lm_new             --numpy.array  (68, 2), y direction is opposite to v direction
        mask_new           --PIL.Image  (target_size, target_size)

    Parameters:
        img                --PIL.Image  (raw_H, raw_W, 3)
        lm                 --numpy.array  (68, 2), y direction is opposite to v direction
        lm3D               --numpy.array  (5, 3)
        mask               --PIL.Image  (raw_H, raw_W, 3)
    """

    w0, h0 = img.size
    if lm.shape[0] != 5:
        lm5p = extract_5p(lm)
    else:
        lm5p = lm

    # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
    t, s = POS(lm5p.transpose(), lm3D.transpose())
    t = t.squeeze()
    s = rescale_factor / s

    # processing the image
    img_new, lm_new, mask_new = resize_n_crop_img(
        img, lm, t, s, target_size=target_size, mask=mask)
    trans_params = np.array([w0, h0, s, t[0], t[1]])

    return trans_params, img_new, lm_new, mask_new


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)[:, None]
    arr /= lens
    return arr


def estimate_normals(vertices, faces):
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    tris = vertices[faces]
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    n[(n[:, 0] == 0) * (n[:, 1] == 0) * (n[:, 2] == 0)] = [0, 0, 1.0]
    n = normalize_v3(n)
    for i in range(3):
        for j in range(faces.shape[0]):
            norm[faces[j, i]] += n[j]

    inds = (norm[:, 0] == 0) * (norm[:, 1] == 0) * (norm[:, 2] == 0)
    norm[inds] = [0, 0, 1.0]
    result = normalize_v3(norm)
    return result


def draw_landmarks(img, landmark, color='r', step=2):
    """
    Return:
        img              -- numpy.array, (B, H, W, 3) img with landmark, RGB order, range (0, 255)


    Parameters:
        img              -- numpy.array, (B, H, W, 3), RGB order, range (0, 255)
        landmark         -- numpy.array, (B, 68, 2), y direction is opposite to v direction
        color            -- str, 'r' or 'b' (red or blue)
    """
    if color == 'r':
        c = np.array([255., 0, 0])
    else:
        c = np.array([0, 0, 255.])

    _, H, W, _ = img.shape
    img, landmark = img.copy(), landmark.copy()
    landmark[..., 1] = H - 1 - landmark[..., 1]
    landmark = np.round(landmark).astype(np.int32)
    for i in range(landmark.shape[1]):
        x, y = landmark[:, i, 0], landmark[:, i, 1]
        for j in range(-step, step):
            for k in range(-step, step):
                u = np.clip(x + j, 0, W - 1)
                v = np.clip(y + k, 0, H - 1)
                for m in range(landmark.shape[0]):
                    img[m, v[m], u[m]] = c
    return img


def split_vis(img_path, target_dir=None):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    n_split = w // h
    if target_dir is None:
        target_dir = os.path.dirname(img_path)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    for i in range(n_split):
        img_i = img[:, i * h:(i + 1) * h, :]
        cv2.imwrite(
            os.path.join(target_dir, '{}_{:0>2d}.jpg'.format(base_name,
                                                             i + 1)), img_i)


def write_video(image_list, save_path, fps=20.0):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # avi格式

    h, w = image_list[0].shape[:2]

    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h), True)

    for frame in image_list:
        out.write(frame)

    out.release()


# ---------------------------- process/generate vertices, normals, faces
def generate_triangles(h, w, margin_x=2, margin_y=5, mask=None):
    # quad layout:
    # 0 1 ... w-1
    # w w+1
    # .
    # w*h
    triangles = []
    for x in range(margin_x, w - 1 - margin_x):
        for y in range(margin_y, h - 1 - margin_y):
            triangle0 = [y * w + x, y * w + x + 1, (y + 1) * w + x]
            triangle1 = [y * w + x + 1, (y + 1) * w + x + 1, (y + 1) * w + x]
            triangles.append(triangle0)
            triangles.append(triangle1)
    triangles = np.array(triangles)
    triangles = triangles[:, [0, 2, 1]]
    return triangles


def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device)
                     * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device)
                     * nv)[:, None, None]  # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.reshape(-1, 3)
    vertices_faces = vertices_faces.reshape(-1, 3, 3)

    normals.index_add_(
        0, faces[:, 1].long(),
        torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1],
                    vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(
        0, faces[:, 2].long(),
        torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2],
                    vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(
        0, faces[:, 0].long(),
        torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0],
                    vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals


def dict2obj(d):
    # if isinstance(d, list):
    #     d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d

    class C(object):
        pass

    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o


def enlarged_bbox(bbox, img_width, img_height, enlarge_ratio=0.2):
    '''
    :param bbox: [xmin,ymin,xmax,ymax]
    :return: bbox: [xmin,ymin,xmax,ymax]
    '''

    left = bbox[0]
    top = bbox[1]

    right = bbox[2]
    bottom = bbox[3]

    roi_width = right - left
    roi_height = bottom - top

    new_left = left - int(roi_width * enlarge_ratio)
    new_left = 0 if new_left < 0 else new_left

    new_top = top - int(roi_height * enlarge_ratio)
    new_top = 0 if new_top < 0 else new_top

    new_right = right + int(roi_width * enlarge_ratio)
    new_right = img_width if new_right > img_width else new_right

    new_bottom = bottom + int(roi_height * enlarge_ratio)
    new_bottom = img_height if new_bottom > img_height else new_bottom

    bbox = [new_left, new_top, new_right, new_bottom]

    bbox = [int(x) for x in bbox]

    return bbox


def draw_line(im, points, color, stroke_size=2, closed=False):
    points = points.astype(np.int32)
    for i in range(len(points) - 1):
        cv2.line(im, tuple(points[i]), tuple(points[i + 1]), color,
                 stroke_size)
    if closed:
        cv2.line(im, tuple(points[0]), tuple(points[-1]), color, stroke_size)
