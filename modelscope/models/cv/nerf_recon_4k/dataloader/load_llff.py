import os

import imageio
import numpy as np
import scipy
import torch


# Slightly modified version of LLFF data loading code
# see https://github.com/Fyusion/LLFF for original
def imread(f):
    if f.endswith('png'):
        return imageio.imread(f, format='PNG-PIL', ignoregamma=True)
    else:
        return imageio.imread(f)


def depthread(path):
    with open(path, 'rb') as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter='&', max_rows=1, usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b'&':
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order='F')
    return np.transpose(array, (1, 0, 2)).squeeze()


def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from shutil import copy
    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [
        f for f in imgs
        if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])
    ]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100. / r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join([
            'mogrify', '-resize', resizearg, '-format', 'png',
            '*.{}'.format(ext)
        ])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def _load_data(basedir,
               factor=None,
               width=None,
               height=None,
               load_imgs=True,
               load_depths=False,
               load_SR=False):

    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    if poses_arr.shape[1] == 17:
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    elif poses_arr.shape[1] == 14:
        poses = poses_arr[:, :-2].reshape([-1, 3, 4]).transpose([1, 2, 0])
    else:
        raise NotImplementedError
    bds = poses_arr[:, -2:].transpose([1, 0])

    img0 = [
        os.path.join(basedir, 'images', f)
        for f in sorted(os.listdir(os.path.join(basedir, 'images')))
        if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
    ][0]
    sh = imageio.imread(img0).shape

    sfx = ''

    if height is not None and width is not None:
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif factor is not None and factor != 1:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'images' + sfx)
    print(f'Loading images from {imgdir}')
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    imgfiles = [
        os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))
        if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
    ]
    if poses.shape[-1] != len(imgfiles):
        print()
        print('Mismatch between imgs {} and poses {} !!!!'.format(
            len(imgfiles), poses.shape[-1]))
        names = set(
            name[:-4]
            for name in np.load(os.path.join(basedir, 'poses_names.npy')))
        assert len(names) == poses.shape[-1]
        print('Below failed files are skip due to SfM failure:')
        new_imgfiles = []
        for i in imgfiles:
            fname = os.path.split(i)[1][:-4]
            if fname in names:
                new_imgfiles.append(i)
            else:
                print('==>', i)
        imgfiles = new_imgfiles

    if len(imgfiles) < 3:
        print('Too few images...')
        import sys
        sys.exit()

    sh = imageio.imread(imgfiles[0]).shape
    if poses.shape[1] == 4:
        poses = np.concatenate([poses, np.zeros_like(poses[:, [0]])], 1)
        poses[2, 4, :] = np.load(os.path.join(basedir, 'hwf_cxcy.npy'))[2]
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor

    if not load_imgs:
        return poses, bds

    imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)

    if load_SR:
        if load_SR == 16:
            imgdir_sr = os.path.join(basedir, 'images_16')
        elif load_SR == 8:
            imgdir_sr = os.path.join(basedir, 'images_8')
        elif load_SR == 4:
            imgdir_sr = os.path.join(basedir, 'images_4')
        elif load_SR == 2:
            imgdir_sr = os.path.join(basedir, 'images_2')
        elif load_SR == 1:
            imgdir_sr = os.path.join(basedir, 'images')
        imgfiles_sr = [
            os.path.join(imgdir_sr, f) for f in sorted(os.listdir(imgdir_sr))
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
        ]
        imgs_sr = [imread(f)[..., :3] / 255. for f in imgfiles_sr]
        imgs_sr = np.stack(imgs_sr, -1)

    print('Loaded image data', imgs.shape, poses[:, -1, 0])

    if not load_depths and load_SR:
        return poses, bds, imgs, imgs_sr

    if not load_depths:
        return poses, bds, imgs

    depthdir = os.path.join(basedir, 'stereo', 'depth_maps')
    assert os.path.exists(depthdir), f'Dir not found: {depthdir}'

    depthfiles = [
        os.path.join(depthdir, f) for f in sorted(os.listdir(depthdir))
        if f.endswith('.geometric.bin')
    ]
    assert poses.shape[-1] == len(
        depthfiles), 'Mismatch between imgs {} and poses {} !!!!'.format(
            len(depthfiles), poses.shape[-1])

    depths = [depthread(f) for f in depthfiles]
    depths = np.stack(depths, -1)
    print('Loaded depth data', depths.shape)
    return poses, bds, imgs, depths


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def w2c_gen(poses):
    final_pose = []
    for idx in range(len(poses)):
        pose = poses[idx, ...]
        z = normalize(pose[:3, 2])
        up = pose[:3, 1]
        vec2 = normalize(z)
        vec0 = normalize(np.cross(up, vec2))
        vec1 = normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2], 1)
        mt = np.linalg.inv(m)
        final_pose.append(mt)
    final_pose = np.stack(final_pose, 0)
    return final_pose


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]

    # -np.sin(theta), -np.sin(theta*zrate)*zdelta
    # 0, 0
    for theta in np.linspace(0., 2 * np.pi * rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([
                np.cos(theta), -np.sin(theta), -np.sin(theta * zrate) * zdelta,
                1.
            ]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def recenter_poses(poses):

    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def rerotate_poses(poses):
    poses = np.copy(poses)
    centroid = poses[:, :3, 3].mean(0)

    poses[:, :3, 3] = poses[:, :3, 3] - centroid

    # Find the minimum pca vector with minimum eigen value
    x = poses[:, :, 3]
    mu = x.mean(0)
    cov = np.cov((x - mu).T)
    ev, eig = np.linalg.eig(cov)
    cams_up = eig[:, np.argmin(ev)]
    if cams_up[1] < 0:
        cams_up = -cams_up

    # Find rotation matrix that align cams_up with [0,1,0]
    R = scipy.spatial.transform.Rotation.align_vectors(
        [[0, 1, 0]], cams_up[None])[0].as_matrix()

    # Apply rotation and add back the centroid position
    poses[:, :3, :3] = R @ poses[:, :3, :3]
    poses[:, :3, [3]] = R @ poses[:, :3, [3]]
    poses[:, :3, 3] = poses[:, :3, 3] + centroid
    return poses


#####################


def spherify_poses(poses, bds, depths):

    def p34_to_44(p):
        return np.concatenate([
            p,
            np.tile(
                np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])
        ], 1)

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv(
            (np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(
        poses[:, :3, :4])

    radius = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1. / radius
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    radius *= sc
    depths *= sc

    poses_reset = np.concatenate([
        poses_reset[:, :3, :4],
        np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)
    ], -1)

    return poses_reset, radius, bds, depths


def load_llff_data(basedir,
                   factor=8,
                   width=None,
                   height=None,
                   recenter=True,
                   rerotate=True,
                   bd_factor=.75,
                   spherify=False,
                   path_zflat=False,
                   load_depths=False,
                   load_SR=False,
                   movie_render_kwargs={}):

    poses, bds, imgs, *depths = _load_data(
        basedir,
        factor=factor,
        width=width,
        height=height,
        load_depths=load_depths,
        load_SR=load_SR)  # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())
    if load_depths:
        depths = depths[0]
    elif load_SR and not load_depths:
        imgs_SRGT = depths[0]
        depths = 0
    else:
        depths = 0

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate(
        [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    if bds.min() < 0 and bd_factor is not None:
        print('Found negative z values from SfM sparse points!?')
        print('Please try bd_factor=None')
        import sys
        sys.exit()
    sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc
    depths *= sc

    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, radius, bds, depths = spherify_poses(poses, bds, depths)
        if rerotate:
            poses = rerotate_poses(poses)

        # generate spiral poses for rendering fly-through movie
        centroid = poses[:, :3, 3].mean(0)
        radcircle = movie_render_kwargs.get('scale_r', 1) * np.linalg.norm(
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
            if movie_render_kwargs.get('flip_up', False):
                up = np.array([0, 1., 0])
            else:
                up = np.array([0, -1., 0])
            vec2 = normalize(camorigin)
            vec0 = normalize(np.cross(vec2, up))
            vec1 = normalize(np.cross(vec2, vec0))
            pos = camorigin + centroid
            # rotate to align with new pitch rotation
            lookat = -vec2
            lookat[1] = target_y
            lookat = normalize(lookat)
            vec2 = -lookat
            vec1 = normalize(np.cross(vec2, vec0))

            p = np.stack([vec0, vec1, vec2, pos], 1)

            render_poses.append(p)

        render_poses = np.stack(render_poses, 0)
        render_poses = np.concatenate([
            render_poses,
            np.broadcast_to(poses[0, :3, -1:], render_poses[:, :3, -1:].shape)
        ], -1)

    else:

        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3, :4])

        # Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
        dt = .75
        mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
        focal = mean_dz * movie_render_kwargs.get('scale_f', 1)

        # Get radii for spiral path
        zdelta = movie_render_kwargs.get('zdelta', 0.5)
        zrate = movie_render_kwargs.get('zrate', 1.0)
        tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0) * movie_render_kwargs.get(
            'scale_r', 1)
        c2w_path = c2w
        N_views = 120
        N_rots = movie_render_kwargs.get('N_rots', 1)
        if path_zflat:
            # zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
            rads[2] = 0.
            N_rots = 1
            N_views /= 2

        # Generate poses for spiral path
        render_poses = render_path_spiral(
            c2w_path,
            up,
            rads,
            focal,
            zdelta,
            zrate=zrate,
            rots=N_rots,
            N=N_views)

    render_poses = torch.Tensor(render_poses)

    # Because both world croodnate system and camera croodnate system are 3-d system, they can be transfer by a:
    # 3x3 rotate matrix and 3x1 moving matrix
    c2w = poses_avg(poses)
    w2c = w2c_gen(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)

    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    if load_SR:
        imgs_SRGT = np.moveaxis(imgs_SRGT, [-1, -2], [0, 1]).astype(np.float32)
    else:
        imgs_SRGT = None

    return images, depths, poses, bds, render_poses, i_test, imgs_SRGT, w2c
