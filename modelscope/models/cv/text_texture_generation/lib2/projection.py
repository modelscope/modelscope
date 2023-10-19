import os
import random
# customized
import sys
from typing import NamedTuple, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch3d.io import save_obj
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (AmbientLights, MeshRasterizer,
                                MeshRendererWithFragments,
                                RasterizationSettings, SoftPhongShader,
                                TexturesUV)
from pytorch3d.renderer.mesh.shader import ShaderBase
from torchvision import transforms
from tqdm import tqdm

from modelscope.models.cv.text_texture_generation.lib2.camera import \
    init_camera
from modelscope.models.cv.text_texture_generation.lib2.init_view import *
from modelscope.models.cv.text_texture_generation.lib2.viusel import (
    visualize_outputs, visualize_quad_mask)

sys.path.append('.')


class BlendParams(NamedTuple):
    sigma: float = 1e-4
    gamma: float = 1e-4
    background_color: Sequence = (1, 1, 1)


class FlatTexelShader(ShaderBase):

    def __init__(self,
                 device='cpu',
                 cameras=None,
                 lights=None,
                 materials=None,
                 blend_params=None):
        super().__init__(device, cameras, lights, materials, blend_params)

    def forward(self, fragments, meshes, **_kwargs):
        texels = meshes.sample_textures(fragments)
        texels[(fragments.pix_to_face == -1), :] = 0
        return texels.squeeze(-2)


def init_soft_phong_shader(camera, blend_params, device):
    lights = AmbientLights(device=device)
    shader = SoftPhongShader(
        cameras=camera,
        lights=lights,
        device=device,
        blend_params=blend_params)

    return shader


def init_flat_texel_shader(camera, device):
    shader = FlatTexelShader(cameras=camera, device=device)
    return shader


def init_renderer(camera, shader, image_size, faces_per_pixel):
    raster_settings = RasterizationSettings(
        image_size=image_size, faces_per_pixel=faces_per_pixel)
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=camera, raster_settings=raster_settings),
        shader=shader)

    return renderer


@torch.no_grad()
def render(mesh, renderer, pad_value=10):

    def phong_normal_shading(meshes, fragments) -> torch.Tensor:
        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        faces_normals = vertex_normals[faces]
        pixel_normals = interpolate_face_attributes(fragments.pix_to_face,
                                                    fragments.bary_coords,
                                                    faces_normals)

        return pixel_normals

    def similarity_shading(meshes, fragments):
        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        faces_normals = vertex_normals[faces]
        vertices = meshes.verts_packed()  # (V, 3)
        face_positions = vertices[faces]
        view_directions = torch.nn.functional.normalize(
            (renderer.shader.cameras.get_camera_center().reshape(1, 1, 3)
             - face_positions),
            p=2,
            dim=2)
        cosine_similarity = torch.nn.CosineSimilarity(dim=2)(faces_normals,
                                                             view_directions)
        pixel_similarity = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords,
            cosine_similarity.unsqueeze(-1))

        return pixel_similarity

    def get_relative_depth_map(fragments, pad_value=pad_value):
        absolute_depth = fragments.zbuf[..., 0]  # B, H, W
        no_depth = -1

        depth_min, depth_max = absolute_depth[absolute_depth != no_depth].min(
        ), absolute_depth[absolute_depth != no_depth].max()
        target_min, target_max = 50, 255

        depth_value = absolute_depth[absolute_depth != no_depth]
        depth_value = depth_max - depth_value  # reverse values

        depth_value /= (depth_max - depth_min)
        depth_value = depth_value * (target_max - target_min) + target_min

        relative_depth = absolute_depth.clone()
        relative_depth[absolute_depth != no_depth] = depth_value
        relative_depth[absolute_depth == no_depth] = pad_value

        return relative_depth

    images, fragments = renderer(mesh)
    normal_maps = phong_normal_shading(mesh, fragments).squeeze(-2)
    similarity_maps = similarity_shading(mesh, fragments).squeeze(-2)  # -1 - 1
    depth_maps = get_relative_depth_map(fragments)

    # normalize similarity mask to 0 - 1
    similarity_maps = torch.abs(similarity_maps)  # 0 - 1

    # HACK erode, eliminate isolated dots
    non_zero_similarity = (similarity_maps > 0).float()
    non_zero_similarity = (non_zero_similarity * 255.).cpu().numpy().astype(
        np.uint8)[0]
    non_zero_similarity = cv2.erode(
        non_zero_similarity, kernel=np.ones((3, 3), np.uint8), iterations=2)
    non_zero_similarity = torch.from_numpy(non_zero_similarity).to(
        similarity_maps.device).unsqueeze(0) / 255.
    similarity_maps = non_zero_similarity.unsqueeze(-1) * similarity_maps
    return images, normal_maps, similarity_maps, depth_maps, fragments


@torch.no_grad()
def check_visible_faces(mesh, fragments):
    pix_to_face = fragments.pix_to_face
    visible_map = pix_to_face.unique()  # (num_visible_faces)
    return visible_map


def get_all_4_locations(values_y, values_x):
    y_0 = torch.floor(values_y)
    y_1 = torch.ceil(values_y)
    x_0 = torch.floor(values_x)
    x_1 = torch.ceil(values_x)

    return torch.cat([y_0, y_0, y_1, y_1],
                     0).long(), torch.cat([x_0, x_1, x_0, x_1], 0).long()


def compose_quad_mask(new_mask_image, update_mask_image, old_mask_image,
                      device):
    """
        compose quad mask:
            -> 0: background
            -> 1: old
            -> 2: update
            -> 3: new
    """

    new_mask_tensor = transforms.ToTensor()(new_mask_image).to(device)
    update_mask_tensor = transforms.ToTensor()(update_mask_image).to(device)
    old_mask_tensor = transforms.ToTensor()(old_mask_image).to(device)

    all_mask_tensor = new_mask_tensor + update_mask_tensor + old_mask_tensor

    quad_mask_tensor = torch.zeros_like(all_mask_tensor)
    quad_mask_tensor[old_mask_tensor == 1] = 1
    quad_mask_tensor[update_mask_tensor == 1] = 2
    quad_mask_tensor[new_mask_tensor == 1] = 3

    return old_mask_tensor, update_mask_tensor, new_mask_tensor, all_mask_tensor, quad_mask_tensor


def compute_view_heat(similarity_tensor, quad_mask_tensor):
    num_total_pixels = quad_mask_tensor.reshape(-1).shape[0]
    heat = 0
    for idx in QUAD_WEIGHTS:
        heat += (quad_mask_tensor
                 == idx).sum() * QUAD_WEIGHTS[idx] / num_total_pixels

    return heat


def select_viewpoint(selected_view_ids,
                     view_punishments,
                     mode,
                     dist_list,
                     elev_list,
                     azim_list,
                     sector_list,
                     view_idx,
                     similarity_texture_cache,
                     exist_texture,
                     mesh,
                     faces,
                     verts_uvs,
                     image_size,
                     faces_per_pixel,
                     init_image_dir,
                     mask_image_dir,
                     normal_map_dir,
                     depth_map_dir,
                     similarity_map_dir,
                     device,
                     use_principle=False):
    if mode == 'sequential':

        num_views = len(dist_list)

        dist = dist_list[view_idx % num_views]
        elev = elev_list[view_idx % num_views]
        azim = azim_list[view_idx % num_views]
        sector = sector_list[view_idx % num_views]

        selected_view_ids.append(view_idx % num_views)

    elif mode == 'heuristic':

        if use_principle and view_idx < 6:

            selected_view_idx = view_idx

        else:

            selected_view_idx = None
            max_heat = 0

            print('=> selecting next view...')
            view_heat_list = []
            for sample_idx in tqdm(range(len(dist_list))):

                view_heat, *_ = render_one_view_and_build_masks(
                    dist_list[sample_idx], elev_list[sample_idx],
                    azim_list[sample_idx], sample_idx, sample_idx,
                    view_punishments, similarity_texture_cache, exist_texture,
                    mesh, faces, verts_uvs, image_size, faces_per_pixel,
                    init_image_dir, mask_image_dir, normal_map_dir,
                    depth_map_dir, similarity_map_dir, device)

                if view_heat > max_heat:
                    selected_view_idx = sample_idx
                    max_heat = view_heat

                view_heat_list.append(view_heat.item())

            print(view_heat_list)
            print('select view {} with heat {}'.format(selected_view_idx,
                                                       max_heat))

        dist = dist_list[selected_view_idx]
        elev = elev_list[selected_view_idx]
        azim = azim_list[selected_view_idx]
        sector = sector_list[selected_view_idx]

        selected_view_ids.append(selected_view_idx)

        view_punishments[selected_view_idx] *= 0.01

    elif mode == 'random':

        selected_view_idx = random.choice(range(len(dist_list)))

        dist = dist_list[selected_view_idx]
        elev = elev_list[selected_view_idx]
        azim = azim_list[selected_view_idx]
        sector = sector_list[selected_view_idx]

        selected_view_ids.append(selected_view_idx)

    else:
        raise NotImplementedError()

    return dist, elev, azim, sector, selected_view_ids, view_punishments


@torch.no_grad()
def build_backproject_mask(mesh, faces, verts_uvs, cameras, reference_image,
                           faces_per_pixel, image_size, uv_size, device):
    # construct pixel UVs
    renderer_scaled = init_renderer(
        cameras,
        shader=init_soft_phong_shader(
            camera=cameras, blend_params=BlendParams(), device=device),
        image_size=image_size,
        faces_per_pixel=faces_per_pixel)
    fragments_scaled = renderer_scaled.rasterizer(mesh)

    # get UV coordinates for each pixel
    faces_verts_uvs = verts_uvs[faces.textures_idx]

    pixel_uvs = interpolate_face_attributes(fragments_scaled.pix_to_face,
                                            fragments_scaled.bary_coords,
                                            faces_verts_uvs)  # NxHsxWsxKx2
    pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(-1, 2)

    texture_locations_y, texture_locations_x = get_all_4_locations(
        (1 - pixel_uvs[:, 1]).reshape(-1) * (uv_size - 1),
        pixel_uvs[:, 0].reshape(-1) * (uv_size - 1))

    K = faces_per_pixel

    texture_values = torch.from_numpy(
        np.array(reference_image.resize(
            (image_size, image_size)))).float() / 255.
    texture_values = texture_values.to(device).unsqueeze(0).expand(
        [4, -1, -1, -1]).unsqueeze(0).expand([K, -1, -1, -1, -1])

    # texture
    texture_tensor = torch.zeros(uv_size, uv_size, 3).to(device)
    texture_tensor[texture_locations_y,
                   texture_locations_x, :] = texture_values.reshape(-1, 3)

    return texture_tensor[:, :, 0]


@torch.no_grad()
def build_diffusion_mask(mesh_stuff,
                         renderer,
                         exist_texture,
                         similarity_texture_cache,
                         target_value,
                         device,
                         image_size,
                         smooth_mask=False,
                         view_threshold=0.01):
    mesh, faces, verts_uvs = mesh_stuff
    mask_mesh = mesh.clone()  # NOTE in-place operation - DANGER!!!

    # visible mask => the whole region
    exist_texture_expand = exist_texture.unsqueeze(0).unsqueeze(-1).expand(
        -1, -1, -1, 3).to(device)
    mask_mesh.textures = TexturesUV(
        maps=torch.ones_like(exist_texture_expand),
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=verts_uvs[None, ...],
        sampling_mode='nearest')
    # visible_mask_tensor, *_ = render(mask_mesh, renderer)
    visible_mask_tensor, _, similarity_map_tensor, *_ = render(
        mask_mesh, renderer)
    # faces that are too rotated away from the viewpoint will be treated as invisible
    valid_mask_tensor = (similarity_map_tensor >= view_threshold).float()
    visible_mask_tensor *= valid_mask_tensor

    # nonexist mask <=> new mask
    exist_texture_expand = exist_texture.unsqueeze(0).unsqueeze(-1).expand(
        -1, -1, -1, 3).to(device)
    mask_mesh.textures = TexturesUV(
        maps=1 - exist_texture_expand,
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=verts_uvs[None, ...],
        sampling_mode='nearest')
    new_mask_tensor, *_ = render(mask_mesh, renderer)
    new_mask_tensor *= valid_mask_tensor

    # exist mask => visible mask - new mask
    exist_mask_tensor = visible_mask_tensor - new_mask_tensor
    exist_mask_tensor[
        exist_mask_tensor < 0] = 0  # NOTE dilate can lead to overflow

    # all update mask
    mask_mesh.textures = TexturesUV(
        maps=(
            similarity_texture_cache.argmax(0) == target_value
            # # only consider the views that have already appeared before
            # similarity_texture_cache[0:target_value+1].argmax(0) == target_value
        ).float().unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, 3).to(device),
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=verts_uvs[None, ...],
        sampling_mode='nearest')
    all_update_mask_tensor, *_ = render(mask_mesh, renderer)

    # current update mask => intersection between all update mask and exist mask
    update_mask_tensor = exist_mask_tensor * all_update_mask_tensor

    # keep mask => exist mask - update mask
    old_mask_tensor = exist_mask_tensor - update_mask_tensor

    # convert
    new_mask = new_mask_tensor[0].cpu().float().permute(2, 0, 1)
    new_mask = transforms.ToPILImage()(new_mask).convert('L')

    update_mask = update_mask_tensor[0].cpu().float().permute(2, 0, 1)
    update_mask = transforms.ToPILImage()(update_mask).convert('L')

    old_mask = old_mask_tensor[0].cpu().float().permute(2, 0, 1)
    old_mask = transforms.ToPILImage()(old_mask).convert('L')

    exist_mask = exist_mask_tensor[0].cpu().float().permute(2, 0, 1)
    exist_mask = transforms.ToPILImage()(exist_mask).convert('L')

    return new_mask, update_mask, old_mask, exist_mask


@torch.no_grad()
def render_one_view(mesh, dist, elev, azim, image_size, faces_per_pixel,
                    device):
    # render the view
    # print(image_size)
    cameras = init_camera(dist, elev, azim, image_size, device)
    renderer = init_renderer(
        cameras,
        shader=init_soft_phong_shader(
            camera=cameras, blend_params=BlendParams(), device=device),
        image_size=image_size,
        faces_per_pixel=faces_per_pixel)

    init_images_tensor, normal_maps_tensor, similarity_tensor, depth_maps_tensor, fragments = render(
        mesh, renderer)
    # print(init_images_tensor.shape, torch.max(init_images_tensor), torch.min(init_images_tensor))
    cv2.imwrite('img.png',
                (np.array(init_images_tensor.squeeze(0)[:, :, :3].cpu())
                 * 255).astype(np.uint8))
    return (cameras, renderer, init_images_tensor, normal_maps_tensor,
            similarity_tensor, depth_maps_tensor, fragments)


@torch.no_grad()
def build_similarity_texture_cache_for_all_views(mesh, faces, verts_uvs,
                                                 dist_list, elev_list,
                                                 azim_list, image_size,
                                                 image_size_scaled, uv_size,
                                                 faces_per_pixel, device):
    num_candidate_views = len(dist_list)
    similarity_texture_cache = torch.zeros(num_candidate_views, uv_size,
                                           uv_size).to(device)

    print('=> building similarity texture cache for all views...')
    for i in tqdm(range(num_candidate_views)):
        cameras, _, _, _, similarity_tensor, _, _ = render_one_view(
            mesh, dist_list[i], elev_list[i], azim_list[i], image_size,
            faces_per_pixel, device)

        similarity_texture_cache[i] = build_backproject_mask(
            mesh, faces, verts_uvs, cameras,
            transforms.ToPILImage()(similarity_tensor[0, :, :,
                                                      0]).convert('RGB'),
            faces_per_pixel, image_size_scaled, uv_size, device)

    return similarity_texture_cache


@torch.no_grad()
def render_one_view_and_build_masks(dist,
                                    elev,
                                    azim,
                                    selected_view_idx,
                                    view_idx,
                                    view_punishments,
                                    similarity_texture_cache,
                                    exist_texture,
                                    mesh,
                                    faces,
                                    verts_uvs,
                                    image_size,
                                    faces_per_pixel,
                                    init_image_dir,
                                    mask_image_dir,
                                    normal_map_dir,
                                    depth_map_dir,
                                    similarity_map_dir,
                                    device,
                                    save_intermediate=False,
                                    smooth_mask=False,
                                    view_threshold=0.01):
    # render the view
    (cameras, renderer, init_images_tensor, normal_maps_tensor,
     similarity_tensor, depth_maps_tensor,
     fragments) = render_one_view(mesh, dist, elev, azim, image_size,
                                  faces_per_pixel, device)

    init_image = init_images_tensor[0].cpu()
    init_image = init_image.permute(2, 0, 1)
    init_image = transforms.ToPILImage()(init_image).convert('RGB')

    normal_map = normal_maps_tensor[0].cpu()
    normal_map = normal_map.permute(2, 0, 1)
    normal_map = transforms.ToPILImage()(normal_map).convert('RGB')

    depth_map = depth_maps_tensor[0].cpu().numpy()
    depth_map = Image.fromarray(depth_map).convert('L')

    similarity_map = similarity_tensor[0, :, :, 0].cpu()
    similarity_map = transforms.ToPILImage()(similarity_map).convert('L')

    flat_renderer = init_renderer(
        cameras,
        shader=init_flat_texel_shader(camera=cameras, device=device),
        image_size=image_size,
        faces_per_pixel=faces_per_pixel)
    new_mask_image, update_mask_image, old_mask_image, exist_mask_image = build_diffusion_mask(
        (mesh, faces, verts_uvs),
        flat_renderer,
        exist_texture,
        similarity_texture_cache,
        selected_view_idx,
        device,
        image_size,
        smooth_mask=smooth_mask,
        view_threshold=view_threshold)
    # NOTE the view idx is the absolute idx in the sample space (i.e. `selected_view_idx`)
    # it should match with `similarity_texture_cache`

    (old_mask_tensor, update_mask_tensor, new_mask_tensor, all_mask_tensor,
     quad_mask_tensor) = compose_quad_mask(new_mask_image, update_mask_image,
                                           old_mask_image, device)

    view_heat = compute_view_heat(similarity_tensor, quad_mask_tensor)
    view_heat *= view_punishments[selected_view_idx]

    # save intermediate results
    if save_intermediate:
        init_image.save(
            os.path.join(init_image_dir, '{}.png'.format(view_idx)))
        normal_map.save(
            os.path.join(normal_map_dir, '{}.png'.format(view_idx)))
        depth_map.save(os.path.join(depth_map_dir, '{}.png'.format(view_idx)))
        similarity_map.save(
            os.path.join(similarity_map_dir, '{}.png'.format(view_idx)))

        new_mask_image.save(
            os.path.join(mask_image_dir, '{}_new.png'.format(view_idx)))
        update_mask_image.save(
            os.path.join(mask_image_dir, '{}_update.png'.format(view_idx)))
        old_mask_image.save(
            os.path.join(mask_image_dir, '{}_old.png'.format(view_idx)))
        exist_mask_image.save(
            os.path.join(mask_image_dir, '{}_exist.png'.format(view_idx)))

        visualize_quad_mask(mask_image_dir, quad_mask_tensor, view_idx,
                            view_heat, device)

    return (view_heat, renderer, cameras, fragments, init_image, normal_map,
            depth_map, init_images_tensor, normal_maps_tensor,
            depth_maps_tensor, similarity_tensor, old_mask_image,
            update_mask_image, new_mask_image, old_mask_tensor,
            update_mask_tensor, new_mask_tensor, all_mask_tensor,
            quad_mask_tensor)


def save_full_obj(output_dir, obj_name, verts, faces, verts_uvs, faces_uvs,
                  projected_texture, device):
    print('=> saving OBJ file...')
    texture_map = transforms.ToTensor()(projected_texture).to(device)
    texture_map = texture_map.permute(1, 2, 0)
    obj_path = os.path.join(output_dir, obj_name)

    save_obj(
        obj_path,
        verts=verts,
        faces=faces,
        decimal_places=5,
        verts_uvs=verts_uvs,
        faces_uvs=faces_uvs,
        texture_map=texture_map)


@torch.no_grad()
def backproject_from_image(mesh, faces, verts_uvs, cameras, reference_image,
                           new_mask_image, update_mask_image, init_texture,
                           exist_texture, image_size, uv_size, faces_per_pixel,
                           device):
    # construct pixel UVs
    renderer_scaled = init_renderer(
        cameras,
        shader=init_soft_phong_shader(
            camera=cameras, blend_params=BlendParams(), device=device),
        image_size=image_size,
        faces_per_pixel=faces_per_pixel)
    fragments_scaled = renderer_scaled.rasterizer(mesh)

    # get UV coordinates for each pixel
    faces_verts_uvs = verts_uvs[faces.textures_idx]

    pixel_uvs = interpolate_face_attributes(fragments_scaled.pix_to_face,
                                            fragments_scaled.bary_coords,
                                            faces_verts_uvs)  # NxHsxWsxKx2
    pixel_uvs = pixel_uvs.permute(0, 3, 1, 2,
                                  4).reshape(pixel_uvs.shape[-2],
                                             pixel_uvs.shape[1],
                                             pixel_uvs.shape[2], 2)

    # the update mask has to be on top of the diffusion mask
    new_mask_image_tensor = transforms.ToTensor()(new_mask_image).to(
        device).unsqueeze(-1)
    update_mask_image_tensor = transforms.ToTensor()(update_mask_image).to(
        device).unsqueeze(-1)

    project_mask_image_tensor = torch.logical_or(
        update_mask_image_tensor, new_mask_image_tensor).float()
    project_mask_image = project_mask_image_tensor * 255.
    project_mask_image = Image.fromarray(
        project_mask_image[0, :, :, 0].cpu().numpy().astype(np.uint8))

    project_mask_image_scaled = project_mask_image.resize(
        (image_size, image_size), )
    # Image.Resampling.NEAREST
    # )
    project_mask_image_tensor_scaled = transforms.ToTensor()(
        project_mask_image_scaled).to(device)

    pixel_uvs_masked = pixel_uvs[project_mask_image_tensor_scaled == 1]

    texture_locations_y, texture_locations_x = get_all_4_locations(
        (1 - pixel_uvs_masked[:, 1]).reshape(-1) * (uv_size - 1),
        pixel_uvs_masked[:, 0].reshape(-1) * (uv_size - 1))

    K = pixel_uvs.shape[0]
    project_mask_image_tensor_scaled = project_mask_image_tensor_scaled[:,
                                                                        None, :, :,
                                                                        None].repeat(
                                                                            1,
                                                                            4,
                                                                            1,
                                                                            1,
                                                                            3)

    texture_values = torch.from_numpy(
        np.array(reference_image.resize((image_size, image_size))))
    texture_values = texture_values.to(device).unsqueeze(0).expand(
        [4, -1, -1, -1]).unsqueeze(0).expand([K, -1, -1, -1, -1])

    texture_values_masked = texture_values.reshape(
        -1, 3)[project_mask_image_tensor_scaled.reshape(-1, 3) == 1].reshape(
            -1, 3)

    # texture
    texture_tensor = torch.from_numpy(np.array(init_texture)).to(device)
    texture_tensor[texture_locations_y,
                   texture_locations_x, :] = texture_values_masked

    init_texture = Image.fromarray(texture_tensor.cpu().numpy().astype(
        np.uint8))

    # update texture cache
    exist_texture[texture_locations_y, texture_locations_x] = 1

    return init_texture, project_mask_image, exist_texture
