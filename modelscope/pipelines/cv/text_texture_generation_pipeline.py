# Copyright © Alibaba, Inc. and its affiliates.
import os
import random
from typing import Any, Dict

import numpy as np
import torch
from diffusers import (ControlNetModel, DiffusionPipeline,
                       EulerAncestralDiscreteScheduler,
                       UniPCMultistepScheduler)
from PIL import Image
from pytorch3d.renderer import TexturesUV
from torchvision import transforms

from modelscope.metainfo import Pipelines
from modelscope.models.cv.text_texture_generation.lib2.camera import *
from modelscope.models.cv.text_texture_generation.lib2.init_view import *
from modelscope.models.cv.text_texture_generation.lib2.projection import *
from modelscope.models.cv.text_texture_generation.lib2.viusel import *
from modelscope.models.cv.text_texture_generation.utils import *
from modelscope.outputs import OutputKeys
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks


@PIPELINES.register_module(
    Tasks.text_texture_generation,
    module_name=Pipelines.text_texture_generation)
class Tex2TexturePipeline(Pipelines):
    """ Stable Diffusion for text_texture_generation Pipeline.
    Example:
    >>> import cv2
    >>> from modelscope.outputs import OutputKeys
    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.utils.constant import Tasks
    >>> input = {'mesh_path':'data/test/mesh/mesh1.obj', 'prompt':'old backpage'}
    >>> model_id = 'damo/cv_diffuser_text-texture-generation'
    >>> txt2texture = pipeline(Tasks.text_texture_generation, model=model_id)
    >>> output = txt2texture(input)
    >>> print(output)
    """

    def __init__(self, model: str, **kwargs):
        super().__init__(model=model, **kwargs)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            print('no gpu avaiable')
            exit()

        enable_xformers_memory_efficient_attention = kwargs.get(
            'enable_xformers_memory_efficient_attention', True)
        try:
            if enable_xformers_memory_efficient_attention:
                self.model.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(e)
        self.model.pipe.enable_model_cpu_offload()
        try:
            if enable_xformers_memory_efficient_attention:
                self.model.inpaintmodel.enable_xformers_memory_efficient_attention(
                )
        except Exception as e:
            print(e)
        self.model.inpaintmodel.enable_model_cpu_offload()

    def preprocess(self, inputs) -> Dict[str, Any]:
        # input: {'mesh_path':'...', 'texture_path':..., uvsize:int, updatestep:int}
        mesh_path = inputs.get('mesh_path', None)
        mesh, verts, faces, aux, mesh_center, scale = self.model.mesh_normalized(
            mesh_path)
        texture_path = inputs.get('texture_path', None)
        prompt = inputs.get('prompt', 'colorful')
        uvsize = inputs.get('uvsize', 1024)
        image_size = inputs.get('image_size', 512)
        output_dir = inputs.get('output_dir', None)
        if texture_path is not None:
            init_texture = Image.open(texture_path).convert('RGB').resize(
                (uvsize, uvsize))
        else:
            zero_map = np.ones((256, 256, 3)) * 127
            init_texture = Image.fromarray(
                zero_map, model='RGB').resize((uvsize, uvsize))
        new_verts_uvs = aux.verts_uvs
        mesh.textures = TexturesUV(
            maps=transforms.ToTensor()(init_texture)[None, ...].permute(
                0, 2, 3, 1).to(self.device),
            faces_uvs=faces.textures_idx[None, ...],
            verts_uvs=new_verts_uvs[None, ...])
        result = {
            'prompt': prompt,
            'mesh': mesh,
            'faces': faces,
            'uvsize': uvsize,
            'mesh_center': mesh_center,
            'scale': scale,
            'verts_uvs': new_verts_uvs,
            'image_size': image_size,
            'init_texture': init_texture,
            'output_dir': output_dir,
        }
        print('mesh load done')
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input['prompt']
        uvsize = input['uvsize']
        mesh = input['mesh']
        mesh_center = input['mesh_center']
        scale = input['scale']
        faces = input['faces']
        verts_uvs = input['verts_uvs']
        image_size = input['image_size']
        init_texture = input['init_texture']
        output_dir = input['output_dir']
        if output_dir is None:
            output_dir = 'Gen_texture'
        exist_texture = torch.from_numpy(
            np.zeros([uvsize, uvsize]).astype(np.float32)).to(self.device)

        generate_dir = os.path.join(output_dir, 'generate')
        os.makedirs(generate_dir, exist_ok=True)

        update_dir = os.path.join(output_dir, 'update')
        os.makedirs(update_dir, exist_ok=True)

        init_image_dir = os.path.join(generate_dir, 'rendering')
        os.makedirs(init_image_dir, exist_ok=True)

        normal_map_dir = os.path.join(generate_dir, 'normal')
        os.makedirs(normal_map_dir, exist_ok=True)

        mask_image_dir = os.path.join(generate_dir, 'mask')
        os.makedirs(mask_image_dir, exist_ok=True)

        depth_map_dir = os.path.join(generate_dir, 'depth')
        os.makedirs(depth_map_dir, exist_ok=True)

        similarity_map_dir = os.path.join(generate_dir, 'similarity')
        os.makedirs(similarity_map_dir, exist_ok=True)

        inpainted_image_dir = os.path.join(generate_dir, 'inpainted')
        os.makedirs(inpainted_image_dir, exist_ok=True)

        mesh_dir = os.path.join(generate_dir, 'mesh')
        os.makedirs(mesh_dir, exist_ok=True)

        interm_dir = os.path.join(generate_dir, 'intermediate')
        os.makedirs(interm_dir, exist_ok=True)

        init_dist = 1.5
        init_elev = 10
        init_azim = 0.0
        fragment_k = 1
        (dist_list, elev_list, azim_list, sector_list,
         view_punishments) = init_viewpoints(
             init_dist, init_elev, init_azim, use_principle=False)
        pre_similarity_texture_cache = build_similarity_texture_cache_for_all_views(
            mesh, faces, verts_uvs, dist_list, elev_list, azim_list,
            image_size, image_size * 8, uvsize, fragment_k, self.device)
        for idx in range(len(dist_list)):
            print('=> processing view {}...'.format(idx))
            dist, elev, azim, sector = dist_list[idx], elev_list[
                idx], azim_list[idx], sector_list[idx]
            prompt_view = ' the {} view of {}'.format(sector, prompt)
            (
                view_score,
                renderer,
                cameras,
                fragments,
                init_image,
                normal_map,
                depth_map,
                init_images_tensor,
                normal_maps_tensor,
                depth_maps_tensor,
                similarity_tensor,
                keep_mask_image,
                update_mask_image,
                generate_mask_image,
                keep_mask_tensor,
                update_mask_tensor,
                generate_mask_tensor,
                all_mask_tensor,
                quad_mask_tensor,
            ) = render_one_view_and_build_masks(
                dist,
                elev,
                azim,
                idx,
                idx,
                view_punishments,
                # => actual view idx and the sequence idx
                pre_similarity_texture_cache,
                exist_texture,
                mesh,
                faces,
                verts_uvs,
                image_size,
                fragment_k,
                init_image_dir,
                mask_image_dir,
                normal_map_dir,
                depth_map_dir,
                similarity_map_dir,
                self.device,
                save_intermediate=True,
                smooth_mask=False,
                view_threshold=0.1)
            generate_image = self.model.pipe(
                prompt_view,
                init_image,
                generate_mask_image,
                depth_maps_tensor,
                strength=1.0)
            init_texture, project_mask_image, exist_texture = backproject_from_image(
                mesh, faces, verts_uvs, cameras, generate_image,
                generate_mask_image, generate_mask_image, init_texture,
                exist_texture, image_size * 8, uvsize, 1, self.device)
            mesh.textures = TexturesUV(
                maps=transforms.ToTensor()(init_texture)[None, ...].permute(
                    0, 2, 3, 1).to(self.device),
                faces_uvs=faces.textures_idx[None, ...],
                verts_uvs=verts_uvs[None, ...])
            (
                view_score,
                renderer,
                cameras,
                fragments,
                init_image,
                *_,
            ) = render_one_view_and_build_masks(
                dist,
                elev,
                azim,
                idx,
                idx,
                view_punishments,
                pre_similarity_texture_cache,
                exist_texture,
                mesh,
                faces,
                verts_uvs,
                image_size,
                8.0,
                init_image_dir,
                mask_image_dir,
                normal_map_dir,
                depth_map_dir,
                similarity_map_dir,
                self.device,
                save_intermediate=False,
                smooth_mask=False,
                view_threshold=0.1)
            if idx > 2:
                diffused_image = self.model.pipe(
                    prompt_view,
                    init_image,
                    update_mask_image,
                    depth_maps_tensor,
                    strength=1.0)
                init_texture, project_mask_image, exist_texture = backproject_from_image(
                    mesh, faces, verts_uvs, cameras, diffused_image,
                    update_mask_image, update_mask_image, init_texture,
                    exist_texture, image_size * 8, uvsize, 1, self.device)
                # update the mesh
                mesh.textures = TexturesUV(
                    maps=transforms.ToTensor()(init_texture)[
                        None, ...].permute(0, 2, 3, 1).to(self.device),
                    faces_uvs=faces.textures_idx[None, ...],
                    verts_uvs=verts_uvs[None, ...])
            inter_images_tensor, *_ = render(mesh, renderer)
            inter_image = inter_images_tensor[0].cpu()
            inter_image = inter_image.permute(2, 0, 1)
            inter_image = transforms.ToPILImage()(inter_image).convert('RGB')
            inter_image.save(os.path.join(interm_dir, '{}.png'.format(idx)))
            exist_texture_image = exist_texture * 255.
            exist_texture_image = Image.fromarray(
                exist_texture_image.cpu().numpy().astype(
                    np.uint8)).convert('L')
            exist_texture_image.save(
                os.path.join(mesh_dir, '{}_texture_mask.png'.format(idx)))

        mask_image = (1 - exist_texture[None, :, :, None])[0].cpu()
        mask_image = mask_image.permute(2, 0, 1)
        mask_image = transforms.ToPILImage()(mask_image).convert('L')
        post_texture = self.model.inpaintmodel(
            prompt=prompt,
            image=init_image.resize((512, 512)),
            mask_image=mask_image.resize((512, 512)),
            height=512,
            width=512).images[0].resize((uvsize, uvsize))
        diffused_image_tensor = torch.from_numpy(np.array(post_texture)).to(
            self.device)
        init_images_tensor = torch.from_numpy(np.array(init_image)).to(
            self.device)
        mask_image_tensor = 1 - exist_texture[None, :, :, None]
        init_images_tensor = diffused_image_tensor * mask_image_tensor[
            0] + init_images_tensor * (1 - mask_image_tensor[0])
        post_texture = Image.fromarray(init_images_tensor.cpu().numpy().astype(
            np.uint8)).convert('RGB')

        save_full_obj(mesh_dir, 'mesh_post.obj',
                      scale * mesh.verts_packed() + mesh_center,
                      faces.verts_idx, verts_uvs, faces.textures_idx,
                      post_texture, self.device)

        return {OutputKeys.OUTPUT: 'Done'}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
