# The implementation is borrowed from https://github.com/YoYo000/MVSNet. Model reading is provided by COLMAP.

from __future__ import print_function
import collections
import multiprocessing as mp
import os
import shutil
import struct
from functools import partial

import cv2
import numpy as np

# ============================ read_model.py ============================#
CameraModel = collections.namedtuple('CameraModel',
                                     ['model_id', 'model_name', 'num_params'])
Camera = collections.namedtuple('Camera',
                                ['id', 'model', 'width', 'height', 'params'])
BaseImage = collections.namedtuple(
    'Image', ['id', 'qvec', 'tvec', 'camera_id', 'name', 'xys', 'point3D_ids'])
Point3D = collections.namedtuple(
    'Point3D', ['id', 'xyz', 'rgb', 'error', 'image_ids', 'point2D_idxs'])


class Image(BaseImage):

    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name='SIMPLE_PINHOLE', num_params=3),
    CameraModel(model_id=1, model_name='PINHOLE', num_params=4),
    CameraModel(model_id=2, model_name='SIMPLE_RADIAL', num_params=4),
    CameraModel(model_id=3, model_name='RADIAL', num_params=5),
    CameraModel(model_id=4, model_name='OPENCV', num_params=8),
    CameraModel(model_id=5, model_name='OPENCV_FISHEYE', num_params=8),
    CameraModel(model_id=6, model_name='FULL_OPENCV', num_params=12),
    CameraModel(model_id=7, model_name='FOV', num_params=5),
    CameraModel(model_id=8, model_name='SIMPLE_RADIAL_FISHEYE', num_params=4),
    CameraModel(model_id=9, model_name='RADIAL_FISHEYE', num_params=5),
    CameraModel(model_id=10, model_name='THIN_PRISM_FISHEYE', num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])


def read_next_bytes(fid,
                    num_bytes,
                    format_char_sequence,
                    endian_character='<'):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_text(path):
    cameras = {}
    with open(path, 'r', encoding='utf-8') as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != '#':
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(
                    id=camera_id,
                    model=model,
                    width=width,
                    height=height,
                    params=params)
    return cameras


def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, 'rb') as fid:
        num_cameras = read_next_bytes(fid, 8, 'Q')[0]
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence='iiQQ')
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid,
                num_bytes=8 * num_params,
                format_char_sequence='d' * num_params)
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_images_text(path):
    images = {}
    with open(path, 'r', encoding='utf-8') as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != '#':
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([
                    tuple(map(float, elems[0::3])),
                    tuple(map(float, elems[1::3]))
                ])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids)
    return images


def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, 'rb') as fid:
        num_reg_images = read_next_bytes(fid, 8, 'Q')[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence='idddddddi')
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ''
            current_char = read_next_bytes(fid, 1, 'c')[0]
            while current_char != b'\x00':  # look for the ASCII 0 entry
                image_name += current_char.decode('utf-8')
                current_char = read_next_bytes(fid, 1, 'c')[0]
            num_points2D = read_next_bytes(
                fid, num_bytes=8, format_char_sequence='Q')[0]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence='ddq' * num_points2D)
            xys = np.column_stack([
                tuple(map(float, x_y_id_s[0::3])),
                tuple(map(float, x_y_id_s[1::3]))
            ])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids)
    return images


def read_points3D_text(path):
    points3D = {}
    with open(path, 'r', encoding='utf-8') as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != '#':
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(
                    id=point3D_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point2D_idxs=point2D_idxs)
    return points3D


def read_points3d_binary(path_to_model_file):
    points3D = {}
    with open(path_to_model_file, 'rb') as fid:
        num_points = read_next_bytes(fid, 8, 'Q')[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence='QdddBBBd')
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence='Q')[0]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence='ii' * track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D


def read_model(path, ext):
    if ext == '.txt':
        cameras = read_cameras_text(os.path.join(path, 'cameras' + ext))
        images = read_images_text(os.path.join(path, 'images' + ext))
        points3D = read_points3D_text(os.path.join(path, 'points3D') + ext)
    else:
        cameras = read_cameras_binary(os.path.join(path, 'cameras' + ext))
        images = read_images_binary(os.path.join(path, 'images' + ext))
        points3D = read_points3d_binary(os.path.join(path, 'points3D') + ext)
    return cameras, images, points3D


def qvec2rotmat(qvec):
    return np.array([
        [
            1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ],  # noqa
        [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ],  # noqa
        [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
        ]
    ])  # noqa


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array(
        [[Rxx - Ryy - Rzz, 0, 0, 0], [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
         [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
         [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0  # noqa
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def calc_score(inputs, images, points3d, extrinsic, args):
    i, j = inputs
    id_i = images[i + 1].point3D_ids
    id_j = images[j + 1].point3D_ids
    id_intersect = [it for it in id_i if it in id_j]
    cam_center_i = -np.matmul(extrinsic[i + 1][:3, :3].transpose(),
                              extrinsic[i + 1][:3, 3:4])[:, 0]
    cam_center_j = -np.matmul(extrinsic[j + 1][:3, :3].transpose(),
                              extrinsic[j + 1][:3, 3:4])[:, 0]
    score = 0
    for pid in id_intersect:
        if pid == -1:
            continue
        p = points3d[pid].xyz
        theta = (180 / np.pi) * np.arccos(
            np.dot(cam_center_i - p, cam_center_j - p)
            / np.linalg.norm(cam_center_i - p)
            / np.linalg.norm(cam_center_j - p))
        tmp_value = (
            2 *  # noqa
            (args.sigma1 if theta <= args.theta0 else args.sigma2)**2)
        score += np.exp(-(theta - args.theta0) *  # noqa
                        (theta - args.theta0) / tmp_value)
    return i, j, score


def processing_single_scene(args):

    image_dir = os.path.join(args.dense_folder, 'images')
    model_dir = os.path.join(args.dense_folder, 'sparse')
    cam_dir = os.path.join(args.save_folder, 'cams')
    image_converted_dir = os.path.join(args.save_folder, 'images_post')

    if os.path.exists(image_converted_dir):
        shutil.rmtree(image_converted_dir)
    os.makedirs(image_converted_dir)
    if os.path.exists(cam_dir):
        shutil.rmtree(cam_dir)

    cameras, images, points3d = read_model(model_dir, args.model_ext)
    num_images = len(list(images.items()))

    param_type = {
        'SIMPLE_PINHOLE': ['f', 'cx', 'cy'],
        'PINHOLE': ['fx', 'fy', 'cx', 'cy'],
        'SIMPLE_RADIAL': ['f', 'cx', 'cy', 'k'],
        'SIMPLE_RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k'],
        'RADIAL': ['f', 'cx', 'cy', 'k1', 'k2'],
        'RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k1', 'k2'],
        'OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2'],
        'OPENCV_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4'],
        'FULL_OPENCV': [
            'fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5',
            'k6'
        ],
        'FOV': ['fx', 'fy', 'cx', 'cy', 'omega'],
        'THIN_PRISM_FISHEYE': [
            'fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1',
            'sy1'
        ]
    }

    # intrinsic
    intrinsic = {}
    for camera_id, cam in cameras.items():
        params_dict = {
            key: value
            for key, value in zip(param_type[cam.model], cam.params)
        }
        if 'f' in param_type[cam.model]:
            params_dict['fx'] = params_dict['f']
            params_dict['fy'] = params_dict['f']
        i = np.array([[params_dict['fx'], 0, params_dict['cx']],
                      [0, params_dict['fy'], params_dict['cy']], [0, 0, 1]])
        intrinsic[camera_id] = i

    new_images = {}
    for i, image_id in enumerate(sorted(images.keys())):
        new_images[i + 1] = images[image_id]
    images = new_images

    # extrinsic
    extrinsic = {}
    for image_id, image in images.items():
        e = np.zeros((4, 4))
        e[:3, :3] = qvec2rotmat(image.qvec)
        e[:3, 3] = image.tvec
        e[3, 3] = 1
        extrinsic[image_id] = e

    # depth range and interval
    depth_ranges = {}
    for i in range(num_images):
        zs = []
        for p3d_id in images[i + 1].point3D_ids:
            if p3d_id == -1:
                continue
            transformed = np.matmul(extrinsic[i + 1], [
                points3d[p3d_id].xyz[0], points3d[p3d_id].xyz[1],
                points3d[p3d_id].xyz[2], 1
            ])
            zs.append(transformed[2].item())
        zs_sorted = sorted(zs)
        # relaxed depth range
        max_ratio = 0.1
        min_ratio = 0.03
        num_max = max(5, int(len(zs) * max_ratio))
        num_min = max(1, int(len(zs) * min_ratio))
        depth_min = 1.0 * sum(zs_sorted[:num_min]) / len(zs_sorted[:num_min])
        depth_max = 1.0 * sum(zs_sorted[-num_max:]) / len(zs_sorted[-num_max:])
        if args.max_d == 0:
            image_int = intrinsic[images[i + 1].camera_id]
            image_ext = extrinsic[i + 1]
            image_r = image_ext[0:3, 0:3]
            image_t = image_ext[0:3, 3]
            p1 = [image_int[0, 2], image_int[1, 2], 1]
            p2 = [image_int[0, 2] + 1, image_int[1, 2], 1]
            P1 = np.matmul(np.linalg.inv(image_int), p1) * depth_min
            P1 = np.matmul(np.linalg.inv(image_r), (P1 - image_t))
            P2 = np.matmul(np.linalg.inv(image_int), p2) * depth_min
            P2 = np.matmul(np.linalg.inv(image_r), (P2 - image_t))
            depth_num = (1 / depth_min - 1 / depth_max) / (
                1 / depth_min - 1 / (depth_min + np.linalg.norm(P2 - P1)))
        else:
            depth_num = args.max_d
        depth_interval = (depth_max - depth_min) / (depth_num
                                                    - 1) / args.interval_scale
        depth_ranges[i + 1] = (depth_min, depth_interval, depth_num, depth_max)

    # view selection
    score = np.zeros((len(images), len(images)))
    queue = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            queue.append((i, j))

    p = mp.Pool(processes=mp.cpu_count())
    func = partial(
        calc_score,
        images=images,
        points3d=points3d,
        args=args,
        extrinsic=extrinsic)
    result = p.map(func, queue)
    for i, j, s in result:
        score[i, j] = s
        score[j, i] = s
    view_sel = []
    for i in range(len(images)):
        sorted_score = np.argsort(score[i])[::-1]
        view_sel.append([(k, score[i, k]) for k in sorted_score[:10]])

    # write
    os.makedirs(cam_dir, exist_ok=True)

    for i in range(num_images):
        with open(os.path.join(cam_dir, '%08d_cam.txt' % i), 'w') as f:
            f.write('extrinsic\n')
            for j in range(4):
                for k in range(4):
                    f.write(str(extrinsic[i + 1][j, k]) + ' ')
                f.write('\n')
            f.write('\nintrinsic\n')
            for j in range(3):
                for k in range(3):
                    f.write(
                        str(intrinsic[images[i + 1].camera_id][j, k]) + ' ')
                f.write('\n')
            f.write('\n%f %f %f %f\n' %
                    (depth_ranges[i + 1][0], depth_ranges[i + 1][1],
                     depth_ranges[i + 1][2], depth_ranges[i + 1][3]))
    with open(os.path.join(args.save_folder, 'pair.txt'), 'w') as f:
        f.write('%d\n' % len(images))
        for i, sorted_score in enumerate(view_sel):
            f.write('%d\n%d ' % (i, len(sorted_score)))
            for image_id, s in sorted_score:
                f.write('%d %f ' % (image_id, s))
            f.write('\n')

    # convert to jpg
    for i in range(num_images):
        img_path = os.path.join(image_dir, images[i + 1].name)
        if not img_path.endswith('.jpg'):
            img = cv2.imread(img_path)
            cv2.imwrite(os.path.join(image_converted_dir, '%08d.jpg' % i), img)
        else:
            shutil.copyfile(
                os.path.join(image_dir, images[i + 1].name),
                os.path.join(image_converted_dir, '%08d.jpg' % i))
