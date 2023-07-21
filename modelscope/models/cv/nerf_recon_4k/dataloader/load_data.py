import numpy as np

from .load_blender import load_blender_data
from .load_llff import load_llff_data
from .load_tankstemple import load_tankstemple_data


def load_data(args):

    K, depths = None, None
    near_clip = None

    if args.dataset_type == 'llff':
        images, depths, poses, bds, render_poses, i_test, *srgt = load_llff_data(
            args.datadir,
            args.factor,
            None,
            None,
            recenter=True,
            bd_factor=0.75,
            spherify=False,
            load_depths=False,
            load_SR=args.load_sr,
            movie_render_kwargs=dict())
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf,
              args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        llffhold = 8
        if llffhold > 0:
            print('Auto LLFF holdout,', llffhold)
            i_test = np.arange(images.shape[0])[::llffhold]

        i_val = [i_test[0]]
        i_train = np.array([
            i for i in np.arange(int(images.shape[0]))
            if (i not in i_test and i not in i_val)
        ])

        print('DEFINING BOUNDS')
        if args.ndc:
            near = 0.
            far = 1.
        else:
            near_clip = max(np.ndarray.min(bds) * .9, 0)
            _far = max(np.ndarray.max(bds) * 1., 0)
            near = 0
            far = inward_nearfar_heuristic(poses[i_train, :3, 3])[1]
            print('near_clip', near_clip)
            print('original far', _far)
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(
            args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf,
              args.datadir)
        i_train, i_val, i_test = i_split

        near, far = 2., 6.

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[..., :3] * images[..., -1:] + (
                    1. - images[..., -1:])
            else:
                images = images[..., :3] * images[..., -1:]

        srgt = [images, 0]

    elif args.dataset_type == 'tankstemple':
        images, poses, render_poses, hwf, K, i_split = load_tankstemple_data(
            args.datadir, movie_render_kwargs=args.movie_render_kwargs)
        print('Loaded tankstemple', images.shape, render_poses.shape, hwf,
              args.datadir)
        i_train, i_val, i_test = i_split

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3], ratio=0)

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[..., :3] * images[..., -1:] + (
                    1. - images[..., -1:])
            else:
                images = images[..., :3] * images[..., -1:]

    else:
        raise NotImplementedError(
            f'Unknown dataset type {args.dataset_type} exiting')

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    HW = np.array([im.shape[:2] for im in images])
    irregular_shape = (images.dtype is np.dtype('object'))

    if K is None:
        K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    render_poses = render_poses[..., :4]

    if args.load_sr:
        srgt, w2c = srgt[0], srgt[1]
    else:
        srgt, w2c = 0, 0

    data_dict = dict(
        hwf=hwf,
        HW=HW,
        Ks=Ks,
        near=near,
        far=far,
        near_clip=near_clip,
        i_train=i_train,
        i_val=i_val,
        i_test=i_test,
        poses=poses,
        render_poses=render_poses,
        images=images,
        depths=depths,
        white_bkgd=args.white_bkgd,
        irregular_shape=irregular_shape,
        srgt=srgt,
        w2c=w2c)
    return data_dict


def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:, None] - cam_o, axis=-1)
    far = dist.max()  # could be too small to exist the scene bbox
    # it is only used to determined scene bbox
    # lib/dvgo use 1e9 as far
    near = far * ratio
    return near, far
