import torch
from kornia import create_meshgrid


def project_and_normalize(ref_grid, src_proj, length):
    """

    @param ref_grid: b 3 n
    @param src_proj: b 4 4
    @param length:   int
    @return:  b, n, 2
    """
    src_grid = src_proj[:, :3, :3] @ ref_grid + src_proj[:, :3, 3:] # b 3 n
    div_val = src_grid[:, -1:]
    div_val[div_val<1e-4] = 1e-4
    src_grid = src_grid[:, :2] / div_val # divide by depth (b, 2, n)
    src_grid[:, 0] = src_grid[:, 0]/((length - 1) / 2) - 1 # scale to -1~1
    src_grid[:, 1] = src_grid[:, 1]/((length - 1) / 2) - 1 # scale to -1~1
    src_grid = src_grid.permute(0, 2, 1) # (b, n, 2)
    return src_grid


def construct_project_matrix(x_ratio, y_ratio, Ks, poses):
    """
    @param x_ratio: float
    @param y_ratio: float
    @param Ks:      b,3,3
    @param poses:   b,3,4
    @return:
    """
    rfn = Ks.shape[0]
    scale_m = torch.tensor([x_ratio, y_ratio, 1.0], dtype=torch.float32, device=Ks.device)
    scale_m = torch.diag(scale_m)
    ref_prj = scale_m[None, :, :] @ Ks @ poses  # rfn,3,4
    pad_vals = torch.zeros([rfn, 1, 4], dtype=torch.float32, device=ref_prj.device)
    pad_vals[:, :, 3] = 1.0
    ref_prj = torch.cat([ref_prj, pad_vals], 1)  # rfn,4,4
    return ref_prj

def get_warp_coordinates(volume_xyz, warp_size, input_size, Ks, warp_pose):
    B, _, D, H, W = volume_xyz.shape
    ratio = warp_size / input_size
    warp_proj = construct_project_matrix(ratio, ratio, Ks, warp_pose) # B,4,4
    warp_coords = project_and_normalize(volume_xyz.view(B,3,D*H*W), warp_proj, warp_size).view(B, D, H, W, 2)
    return warp_coords


def create_target_volume(depth_size, volume_size, input_image_size, pose_target, K, near=None, far=None):
    device, dtype = pose_target.device, pose_target.dtype

    # compute a depth range on the unit sphere
    H, W, D, B = volume_size, volume_size, depth_size, pose_target.shape[0]
    if near is not None and far is not None :
        # near, far b,1,h,w
        depth_values = torch.linspace(0, 1, steps=depth_size).to(near.device).to(near.dtype) # d
        depth_values = depth_values.view(1, D, 1, 1) # 1,d,1,1
        depth_values = depth_values * (far - near) + near # b d h w
        depth_values = depth_values.view(B, 1, D, H * W)
    else:
        near, far = near_far_from_unit_sphere_using_camera_poses(pose_target) # b 1
        depth_values = torch.linspace(0, 1, steps=depth_size).to(near.device).to(near.dtype) # d
        depth_values = depth_values[None,:,None] * (far[:,None,:] - near[:,None,:]) + near[:,None,:] # b d 1
        depth_values = depth_values.view(B, 1, D, 1).expand(B, 1, D, H*W)

    ratio = volume_size / input_image_size

    # creat a grid on the target (reference) view
    # H, W, D, B = volume_size, volume_size, depth_values.shape[1], depth_values.shape[0]

    # creat mesh grid: note reference also means target
    ref_grid = create_meshgrid(H, W, normalized_coordinates=False)  # (1, H, W, 2)
    ref_grid = ref_grid.to(device).to(dtype)
    ref_grid = ref_grid.permute(0, 3, 1, 2) # (1, 2, H, W)
    ref_grid = ref_grid.reshape(1, 2, H*W) # (1, 2, H*W)
    ref_grid = ref_grid.expand(B, -1, -1) # (B, 2, H*W)
    ref_grid = torch.cat((ref_grid, torch.ones(B, 1, H*W, dtype=ref_grid.dtype, device=ref_grid.device)), dim=1) # (B, 3, H*W)
    ref_grid = ref_grid.unsqueeze(2) * depth_values  # (B, 3, D, H*W)

    # unproject to space and transfer to world coordinates.
    Ks = K
    ref_proj = construct_project_matrix(ratio, ratio, Ks, pose_target) # B,4,4
    ref_proj_inv = torch.inverse(ref_proj) # B,4,4
    ref_grid = ref_proj_inv[:,:3,:3] @ ref_grid.view(B,3,D*H*W) + ref_proj_inv[:,:3,3:] # B,3,3 @ B,3,DHW + B,3,1 => B,3,DHW
    return ref_grid.reshape(B,3,D,H,W), depth_values.view(B,1,D,H,W)

def near_far_from_unit_sphere_using_camera_poses(camera_poses):
    """
    @param camera_poses: b 3 4
    @return:
        near: b,1
        far: b,1
    """
    R_w2c = camera_poses[..., :3, :3] # b 3 3
    t_w2c = camera_poses[..., :3, 3:] # b 3 1
    camera_origin = -R_w2c.permute(0,2,1) @ t_w2c   # b 3 1
    # R_w2c.T @ (0,0,1) = z_dir
    camera_orient = R_w2c.permute(0,2,1)[...,:3,2:3] # b 3 1
    camera_origin, camera_orient = camera_origin[...,0], camera_orient[..., 0] # b 3
    a = torch.sum(camera_orient ** 2, dim=-1, keepdim=True) # b 1
    b = -torch.sum(camera_orient * camera_origin, dim=-1, keepdim=True) # b 1
    mid = b / a # b 1
    near, far = mid - 1.0, mid + 1.0
    return near, far