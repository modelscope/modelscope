import glob
import os

import imageio
import numpy as np


def normalize(x):
    return x / np.linalg.norm(x)


def load_tankstemple_data(basedir, movie_render_kwargs={}):
    pose_paths = sorted(glob.glob(os.path.join(basedir, 'pose', '*txt')))
    rgb_paths = sorted(glob.glob(os.path.join(basedir, 'rgb', '*png')))

    all_poses = []
    all_imgs = []
    i_split = [[], []]
    for i, (pose_path, rgb_path) in enumerate(zip(pose_paths, rgb_paths)):
        i_set = int(os.path.split(rgb_path)[-1][0])
        all_poses.append(np.loadtxt(pose_path).astype(np.float32))
        all_imgs.append((imageio.imread(rgb_path) / 255.).astype(np.float32))
        i_split[i_set].append(i)

    imgs = np.stack(all_imgs, 0)
    poses = np.stack(all_poses, 0)
    i_split.append(i_split[-1])

    path_intrinsics = os.path.join(basedir, 'intrinsics.txt')
    H, W = imgs[0].shape[:2]
    K = np.loadtxt(path_intrinsics)
    focal = float(K[0, 0])

    # generate spiral poses for rendering fly-through movie
    centroid = poses[:, :3, 3].mean(0)
    radcircle = movie_render_kwargs.get('scale_r', 1.0) * np.linalg.norm(
        poses[:, :3, 3] - centroid, axis=-1).mean()
    centroid[0] += movie_render_kwargs.get('shift_x', 0)
    centroid[1] += movie_render_kwargs.get('shift_y', 0)
    centroid[2] += movie_render_kwargs.get('shift_z', 0)
    new_up_rad = movie_render_kwargs.get('pitch_deg', 0) * np.pi / 180
    target_y = radcircle * np.tan(new_up_rad)

    render_poses = []

    for th in np.linspace(0., 2. * np.pi, 200):
        camorigin = np.array(
            [radcircle * np.cos(th), 0, radcircle * np.sin(th)])
        if movie_render_kwargs.get('flip_up_vec', False):
            up = np.array([0, -1., 0])
        else:
            up = np.array([0, 1., 0])
        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin + centroid
        # rotate to align with new pitch rotation
        lookat = -vec2
        lookat[1] = target_y
        lookat = normalize(lookat)
        lookat *= -1
        vec2 = -lookat
        vec1 = normalize(np.cross(vec2, vec0))

        p = np.stack([vec0, vec1, vec2, pos], 1)

        render_poses.append(p)

    render_poses = np.stack(render_poses, 0)
    render_poses = np.concatenate([
        render_poses,
        np.broadcast_to(poses[0, :3, -1:], render_poses[:, :3, -1:].shape)
    ], -1)

    return imgs, poses, render_poses, [H, W, focal], K, i_split
