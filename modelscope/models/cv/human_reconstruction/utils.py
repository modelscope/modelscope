import os

import mcubes
import numpy as np
import torch


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')
    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' %
                   (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')
    for idx, v in enumerate(verts):
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def to_tensor(img):
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    img = torch.from_numpy(img.transpose(2, 0, 1)).float()
    img = img / 255.
    return img


def reconstruction(net, calib_tensor, coords, mat, num_samples=50000):

    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        points = np.repeat(points, 1, axis=0)
        samples = torch.from_numpy(points).cuda().float()
        net.query(samples, calib_tensor)
        pred = net.get_preds()
        pred = pred[0]
        return pred.detach().cpu().numpy()

    sdf = eval_grid(coords, eval_func, num_samples=num_samples)
    vertices, faces = mcubes.marching_cubes(sdf, 0.5)
    verts = np.matmul(mat[:3, :3], vertices.T) + mat[:3, 3:4]
    verts = verts.T
    return verts, faces


def keep_largest(mesh_big):
    mesh_lst = mesh_big.split(only_watertight=False)
    keep_mesh = mesh_lst[0]
    for mesh in mesh_lst:
        if mesh.vertices.shape[0] > keep_mesh.vertices.shape[0]:
            keep_mesh = mesh
    return keep_mesh


def eval_grid(coords,
              eval_func,
              init_resolution=64,
              threshold=0.01,
              num_samples=512 * 512 * 512):
    resolution = coords.shape[1:4]
    sdf = np.zeros(resolution)
    dirty = np.ones(resolution, dtype=bool)
    grid_mask = np.zeros(resolution, dtype=bool)
    reso = resolution[0] // init_resolution

    while reso > 0:
        grid_mask[0:resolution[0]:reso, 0:resolution[1]:reso,
                  0:resolution[2]:reso] = True
        test_mask = np.logical_and(grid_mask, dirty)
        points = coords[:, test_mask]

        sdf[test_mask] = batch_eval(points, eval_func, num_samples=num_samples)
        dirty[test_mask] = False

        if reso <= 1:
            break
        for x in range(0, resolution[0] - reso, reso):
            for y in range(0, resolution[1] - reso, reso):
                for z in range(0, resolution[2] - reso, reso):
                    if not dirty[x + reso // 2, y + reso // 2, z + reso // 2]:
                        continue
                    v0 = sdf[x, y, z]
                    v1 = sdf[x, y, z + reso]
                    v2 = sdf[x, y + reso, z]
                    v3 = sdf[x, y + reso, z + reso]
                    v4 = sdf[x + reso, y, z]
                    v5 = sdf[x + reso, y, z + reso]
                    v6 = sdf[x + reso, y + reso, z]
                    v7 = sdf[x + reso, y + reso, z + reso]
                    v = np.array([v0, v1, v2, v3, v4, v5, v6, v7])
                    v_min = v.min()
                    v_max = v.max()
                    if (v_max - v_min) < threshold:
                        sdf[x:x + reso, y:y + reso,
                            z:z + reso] = (v_max + v_min) / 2
                        dirty[x:x + reso, y:y + reso, z:z + reso] = False
        reso //= 2

    return sdf.reshape(resolution)


def batch_eval(points, eval_func, num_samples=512 * 512 * 512):
    num_pts = points.shape[1]
    sdf = np.zeros(num_pts)

    num_batches = num_pts // num_samples
    for i in range(num_batches):
        sdf[i * num_samples:i * num_samples + num_samples] = eval_func(
            points[:, i * num_samples:i * num_samples + num_samples])
    if num_pts % num_samples:
        sdf[num_batches * num_samples:] = eval_func(points[:, num_batches
                                                           * num_samples:])
    return sdf


def create_grid(res,
                b_min=np.array([0, 0, 0]),
                b_max=np.array([1, 1, 1]),
                transform=None):
    coords = np.mgrid[:res, :res, :res]

    coords = coords.reshape(3, -1)
    coords_matrix = np.eye(4)
    length = b_max - b_min

    coords_matrix[0, 0] = length[0] / res
    coords_matrix[1, 1] = length[1] / res
    coords_matrix[2, 2] = length[2] / res
    coords_matrix[0:3, 3] = b_min

    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    if transform is not None:
        coords = np.matmul(transform[:3, :3], coords) + transform[:3, 3:4]
        coords_matrix = np.matmul(transform, coords_matrix)
    coords = coords.reshape(3, res, res, res)
    return coords, coords_matrix


def get_submesh(verts,
                faces,
                color,
                verts_retained=None,
                faces_retained=None,
                min_vert_in_face=2):
    verts = verts
    faces = faces
    colors = color
    if verts_retained is not None:
        if verts_retained.dtype != 'bool':
            vert_mask = np.zeros(len(verts), dtype=bool)
            vert_mask[verts_retained] = True
        else:
            vert_mask = verts_retained
        bool_faces = np.sum(
            vert_mask[faces.ravel()].reshape(-1, 3), axis=1) > min_vert_in_face
    elif faces_retained is not None:
        if faces_retained.dtype != 'bool':
            bool_faces = np.zeros(len(faces_retained), dtype=bool)
        else:
            bool_faces = faces_retained
    new_faces = faces[bool_faces]
    vertex_ids = list(set(new_faces.ravel()))
    oldtonew = -1 * np.ones([len(verts)])
    oldtonew[vertex_ids] = range(0, len(vertex_ids))
    new_verts = verts[vertex_ids]
    new_colors = colors[vertex_ids]
    new_faces = oldtonew[new_faces].astype('int32')
    return (new_verts, new_faces, new_colors, bool_faces, vertex_ids)
