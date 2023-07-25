import argparse
import os
import random
import time

import imageio
import mmcv
import numpy as np
import torch
from tqdm import tqdm, trange

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .dataloader.load_data import load_data
from .network.dvgo import DirectMPIGO, DirectVoxGO, SFTNet, get_rays_of_a_view

logger = get_logger()


def to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


__all__ = ['NeRFRecon4K']


@MODELS.register_module(Tasks.nerf_recon_4k, module_name=Models.nerf_recon_4k)
class NeRFRecon4K(TorchModel):

    def __init__(self, model_dir, **kwargs):
        super().__init__(model_dir, **kwargs)

        if not torch.cuda.is_available():
            raise Exception('GPU is required')
        self.device = torch.device('cuda')
        logger.info('model params:{}'.format(kwargs))
        self.data_type = kwargs['data_type']
        # self.use_mask = kwargs['use_mask']
        # self.num_samples_per_ray = kwargs['num_samples_per_ray']
        self.test_ray_chunk = kwargs['test_ray_chunk']
        # self.enc_ckpt_path = kwargs['enc_ckpt_path']
        # self.dec_ckpt_path = kwargs['dec_ckpt_path']

        self.enc_ckpt_path = os.path.join(model_dir, 'fine_100000.tar')
        if not os.path.exists(self.enc_ckpt_path):
            raise Exception('encoder ckpt path not found')
        # if self.dec_ckpt_path == '':
        self.dec_ckpt_path = os.path.join(model_dir, 'sresrnet_100000.pth')
        if not os.path.exists(self.dec_ckpt_path):
            raise Exception('decoder ckpt path not found')

        self.ckpt_name = self.dec_ckpt_path.split('/')[-1][:-4]
        self.ndc = True if self.data_type == 'llff' else False
        self.sr_ratio = int(kwargs['factor'] / kwargs['load_sr'])
        self.load_existed_model()

        self.test_tile = kwargs['test_tile']
        self.stepsize = kwargs['stepsize']

    def load_existed_model(self):
        if self.ndc:
            model_class = DirectMPIGO
            ckpt = torch.load(self.enc_ckpt_path, map_location='cpu')
        else:
            model_class = DirectVoxGO
            ckpt = torch.load(self.enc_ckpt_path, map_location='cpu')
            ckpt['model_kwargs']['mask_cache_path'] = self.enc_ckpt_path
        self.encoder = model_class(**ckpt['model_kwargs'])
        self.encoder.load_state_dict(ckpt['model_state_dict'])
        self.encoder = self.encoder.to(self.device)

        self.decoder = SFTNet(
            n_in_colors=3,
            scale=self.sr_ratio,
            num_feat=64,
            num_block=5,
            num_grow_ch=32,
            num_cond=1,
            dswise=False).to(self.device)
        self.decoder.load_network(
            load_path=self.dec_ckpt_path, device=self.device)
        self.decoder.eval()

    def nerf_reconstruction(self, data_cfg, render_dir):
        data_dict = load_everything(cfg_data=data_cfg)

        self.render_viewpoints_kwargs = {
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if data_dict['white_bkgd'] else 0,
                'stepsize': self.stepsize,
                'inverse_y': False,
                'flip_x': False,
                'flip_y': False,
                'render_depth': True,
            },
        }

        os.makedirs(render_dir, exist_ok=True)
        print('All results are dumped into', render_dir)
        rgbs, depths, bgmaps, _, _, rgb_features = self.render_viewpoints(
            render_poses=data_dict['poses'][data_dict['i_test']],
            HW=data_dict['HW'][data_dict['i_test']],
            Ks=data_dict['Ks'][data_dict['i_test']],
            gt_imgs=[
                data_dict['images'][i].cpu().numpy()
                for i in data_dict['i_test']
            ],
            savedir=render_dir,
            dump_images=False,
            **self.render_viewpoints_kwargs)

        rgbsr = []
        for idx, rgbsave in enumerate(tqdm(rgb_features)):
            rgbtest = torch.from_numpy(rgbsave).movedim(-1, 0).unsqueeze(0).to(
                self.device)
            # rgb = torch.from_numpy(rgbs[idx]).movedim(-1, 0).unsqueeze(0).to(self.device)

            input_cond = torch.from_numpy(depths).movedim(-1, 1)
            input_cond = input_cond[idx, :, :, :].to(self.device)

            if self.test_tile:
                rgb_srtest = self.decoder.tile_process(
                    rgbtest, input_cond, tile_size=self.test_tile)
            else:
                rgb_srtest = self.decoder(rgbtest,
                                          input_cond).detach().to('cpu')

            rgb_srsave = rgb_srtest.squeeze().movedim(0, -1).detach().clamp(
                0, 1).numpy()
            rgbsr.append(rgb_srsave)
        print(
            '''all inference process has done, saving images... because our images are
            4K (4032x3024), the saving process may be time-consuming.''')
        rgbsr = np.array(rgbsr)
        for i in trange(len(rgbsr)):
            rgb8 = to8b(rgbsr[i])
            filename = os.path.join(render_dir, '{:03d}_dec.png'.format(i))
            imageio.imwrite(filename, rgb8)

        imageio.mimwrite(
            os.path.join(render_dir, 'result_dec.mp4'),
            to8b(rgbsr),
            fps=25,
            codec='libx264',
            quality=8)

    @torch.no_grad()
    def render_viewpoints(self,
                          render_poses,
                          HW,
                          Ks,
                          render_kwargs,
                          gt_imgs=None,
                          savedir=None,
                          dump_images=False,
                          render_factor=0,
                          eval_ssim=False,
                          eval_lpips_alex=False,
                          eval_lpips_vgg=False):
        '''Render images for the given viewpoints; run evaluation if gt given.
        '''
        assert len(render_poses) == len(HW) and len(HW) == len(Ks)

        if render_factor != 0:
            HW = np.copy(HW)
            Ks = np.copy(Ks)
            HW = (HW / render_factor).astype(int)
            Ks[:, :2, :3] /= render_factor

        rgbs = []
        rgb_features = []
        depths = []
        bgmaps = []
        psnrs = []
        viewdirs_all = []
        ssims = []
        lpips_alex = []
        lpips_vgg = []

        for i, c2w in enumerate(tqdm(render_poses)):

            H, W = HW[i]
            K = Ks[i]
            c2w = torch.Tensor(c2w)
            rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H,
                W,
                K,
                c2w,
                self.ndc,
                inverse_y=False,
                flip_x=False,
                flip_y=False)
            keys = ['rgb_marched', 'depth', 'alphainv_last', 'rgb_feature']
            rays_o = rays_o.flatten(0, -2).to('cuda')
            rays_d = rays_d.flatten(0, -2).to('cuda')
            viewdirs = viewdirs.flatten(0, -2).to('cuda')
            time_rdstart = time.time()
            render_result_chunks = [{
                k: v
                for k, v in self.encoder(ro, rd, vd, **render_kwargs).items()
                if k in keys
            } for ro, rd, vd in zip(
                rays_o.split(self.test_ray_chunk, 0),
                rays_d.split(self.test_ray_chunk, 0),
                viewdirs.split(self.test_ray_chunk, 0))]
            render_result = {
                k:
                torch.cat([ret[k]
                           for ret in render_result_chunks]).reshape(H, W, -1)
                for k in render_result_chunks[0].keys()
            }
            print(f'render time is: {time.time() - time_rdstart}')
            rgb = render_result['rgb_marched'].clamp(0, 1).cpu().numpy()
            rgb_feature = render_result['rgb_feature'].cpu().numpy()
            depth = render_result['depth'].cpu().numpy()
            bgmap = render_result['alphainv_last'].cpu().numpy()

            rgbs.append(rgb)
            rgb_features.append(rgb_feature)
            depths.append(depth)
            bgmaps.append(bgmap)
            viewdirs_all.append(viewdirs)
            if i == 0:
                print('Testing', rgb.shape)

            if gt_imgs is not None and render_factor == 0:
                p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
                psnrs.append(p)

        if len(psnrs):
            print('Testing psnr', np.mean(psnrs), '(avg)')
            if eval_ssim:
                print('Testing ssim', np.mean(ssims), '(avg)')
            if eval_lpips_vgg:
                print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
            if eval_lpips_alex:
                print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

        if savedir is not None and dump_images:
            for i in trange(len(rgbs)):
                rgb8 = to8b(rgbs[i])
                filename = os.path.join(savedir, '{:03d}_enc.png'.format(i))
                imageio.imwrite(filename, rgb8)

        rgbs = np.array(rgbs)
        rgb_features = np.array(rgb_features)
        depths = np.array(depths)
        bgmaps = np.array(bgmaps)

        return rgbs, depths, bgmaps, psnrs, viewdirs_all, rgb_features


def load_everything(cfg_data):
    '''Load images / poses / camera settings / data split.
    '''
    cfg_data = mmcv.Config(cfg_data)
    data_dict = load_data(cfg_data)

    # remove useless field
    kept_keys = {
        'hwf', 'HW', 'Ks', 'near', 'far', 'near_clip', 'i_train', 'i_val',
        'i_test', 'irregular_shape', 'poses', 'render_poses', 'images',
        'white_bkgd'
    }
    # if cfg.data.load_sr:
    kept_keys.add('srgt')
    kept_keys.add('w2c')
    data_dict['srgt'] = torch.FloatTensor(data_dict['srgt'], device='cpu')
    data_dict['w2c'] = torch.FloatTensor(data_dict['w2c'], device='cpu')
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [
            torch.FloatTensor(im, device='cpu') for im in data_dict['images']
        ]
    else:
        data_dict['images'] = torch.FloatTensor(
            data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict
