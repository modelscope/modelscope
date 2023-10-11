# common utils
import os

import imageio.v2 as imageio
import torch
# pytorch3d
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.renderer import (AmbientLights, MeshRasterizer,
                                MeshRendererWithFragments, PerspectiveCameras,
                                RasterizationSettings, SoftPhongShader,
                                look_at_view_transform)
from torchvision import transforms
from tqdm import tqdm

IMAGE_SIZE = 768


def init_mesh(model_path, device):
    verts, faces, aux = load_obj(model_path, device=device)
    mesh = load_objs_as_meshes([model_path], device=device)
    return mesh, verts, faces, aux


def init_camera(num_views, dist, elev, azim, view_idx, device):
    interval = 360 // num_views
    azim = (azim + interval * view_idx) % 360
    R, T = look_at_view_transform(dist, elev, azim)
    T[0][2] = dist
    image_size = torch.tensor([IMAGE_SIZE, IMAGE_SIZE]).unsqueeze(0)
    focal_length = torch.tensor(2.0)
    cameras = PerspectiveCameras(
        focal_length=focal_length,
        R=R,
        T=T,
        device=device,
        image_size=image_size)
    return cameras, dist, elev, azim


def init_renderer(camera, device):
    raster_settings = RasterizationSettings(image_size=IMAGE_SIZE)
    lights = AmbientLights(device=device)
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=camera, raster_settings=raster_settings),
        shader=SoftPhongShader(cameras=camera, lights=lights, device=device))

    return renderer


def generation_gif(mesh_path):
    num_views = 72
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
        torch.cuda.set_device(DEVICE)
    else:
        print('no gpu avaiable')
        exit()
    output_dir = 'GIF-{}'.format(num_views)
    os.makedirs(output_dir, exist_ok=True)

    mesh, verts, faces, aux = init_mesh(mesh_path, DEVICE)

    # rendering
    print('=> rendering...')
    for view_idx in tqdm(range(num_views)):
        init_image_path = os.path.join(output_dir, '{}.png'.format(view_idx))
        dist = 1.8
        elev = 15
        azim = 0

        cameras, dist, elev, azim = init_camera(num_views, dist, elev, azim,
                                                view_idx, DEVICE)
        renderer = init_renderer(cameras, DEVICE)
        init_images_tensor, fragments = renderer(mesh)

        # save images
        init_image = init_images_tensor[0].cpu()
        init_image = init_image.permute(2, 0, 1)
        init_image = transforms.ToPILImage()(init_image).convert('RGB')
        init_image.save(init_image_path)

    # generate GIF
    images = [
        imageio.imread(os.path.join(output_dir, '{}.png').format(v_id))
        for v_id in range(args.num_views)
    ]
    imageio.mimsave(
        os.path.join(output_dir, 'output.gif'), images, duration=0.1)
    imageio.mimsave(os.path.join(output_dir, 'output.mp4'), images, fps=25)
    print('=> done!')
