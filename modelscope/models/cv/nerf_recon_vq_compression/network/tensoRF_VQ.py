import os
import random
from typing import Callable, Iterator, List, Optional, Union

import torch.nn as nn
from tqdm import tqdm

from .tensorBase import *
from .tensoRF import TensorVMSplit
from .weighted_vq import VectorQuantize


class Timing:
    """
    Timing environment
    usage:
    with Timing("message"):
        your commands here
    will print CUDA runtime in ms
    """

    def __init__(self, name, debug=False):
        self.name = name
        self.debug = debug

    def __enter__(self):
        if not self.debug:
            return

        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def __exit__(self, type, value, traceback):
        if not self.debug:
            return

        self.end.record()
        torch.cuda.synchronize()
        print(self.name, 'elapsed', self.start.elapsed_time(self.end), 'ms')


def dec2bin(x, bits):
    mask = 2**torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


def bin2dec(b, bits):
    mask = 2**torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)


class TensorVMSplitVQ(TensorVMSplit):

    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorVMSplitVQ, self).__init__(aabb, gridSize, device, **kargs)
        self.codebook_size = kargs['codebook_size']
        print('codebook size: ' + str(self.codebook_size))
        self.use_cosine_sim = kargs['use_cosine_sim'] == 1
        self.codebook_dim = None if kargs['codebook_dim'] == 0 else kargs[
            'codebook_dim']
        self.vq = nn.ModuleList([
            VectorQuantize(
                dim=self.app_n_comp[0],
                codebook_size=self.codebook_size,  # codebook size
                decay=0.8,  # specify number of quantizer
                commitment_weight=1.0,
                use_cosine_sim=self.use_cosine_sim,
                codebook_dim=self.codebook_dim,
                threshold_ema_dead_code=2.0,
            ).to(self.device),
            VectorQuantize(
                dim=self.app_n_comp[1],
                codebook_size=self.codebook_size,  # codebook size
                decay=0.8,  # specify number of quantizer
                commitment_weight=1.0,
                use_cosine_sim=self.use_cosine_sim,
                codebook_dim=self.codebook_dim,
                threshold_ema_dead_code=2.0,
            ).to(self.device),
            VectorQuantize(
                dim=self.app_n_comp[2],
                codebook_size=self.codebook_size,  # codebook size
                decay=0.8,  # specify number of quantizer
                commitment_weight=1.0,
                use_cosine_sim=self.use_cosine_sim,
                codebook_dim=self.codebook_dim,
                threshold_ema_dead_code=2.0,
            ).to(self.device)
        ])
        self.den_vq = nn.ModuleList([
            VectorQuantize(
                dim=self.density_n_comp[0],
                codebook_size=self.codebook_size,  # codebook size
                decay=0.8,  # specify number of quantizer
                commitment_weight=1.0,
                use_cosine_sim=self.use_cosine_sim,
                codebook_dim=self.codebook_dim,
                threshold_ema_dead_code=2.0,
            ).to(self.device),
            VectorQuantize(
                dim=self.density_n_comp[1],
                codebook_size=self.codebook_size,  # codebook size
                decay=0.8,  # specify number of quantizer
                commitment_weight=1.0,
                use_cosine_sim=self.use_cosine_sim,
                codebook_dim=self.codebook_dim,
                threshold_ema_dead_code=2.0,
            ).to(self.device),
            VectorQuantize(
                dim=self.density_n_comp[2],
                codebook_size=self.codebook_size,  # codebook size
                decay=0.8,  # specify number of quantizer
                commitment_weight=1.0,
                use_cosine_sim=self.use_cosine_sim,
                codebook_dim=self.codebook_dim,
                threshold_ema_dead_code=2.0,
            ).to(self.device)
        ])
        self.importance = kargs.get('importance', None)
        self.plane_mask = kargs.get('plane_mask', None)
        self.all_indices = kargs.get('all_indices', None)

    def compute_appfeature_vq(self, xyz_sampled, is_train):

        # plane + line basis
        coordinate_plane = torch.stack(
            (xyz_sampled[..., self.matMode[0]], xyz_sampled[...,
                                                            self.matMode[1]],
             xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack(
            (xyz_sampled[..., self.vecMode[0]],
             xyz_sampled[..., self.vecMode[1]], xyz_sampled[...,
                                                            self.vecMode[2]]))
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line),
            dim=-1).detach().view(3, -1, 1, 2)  # (3*B*1*2)

        plane_coef_point, line_coef_point = [], []
        loss = 0
        for idx_plane in range(len(self.app_plane)):  #app_plane [48*128*128]
            feat_after_tri_T = F.grid_sample(
                self.app_plane[idx_plane],
                coordinate_plane[[idx_plane]],
                align_corners=True).view(-1, *xyz_sampled.shape[:1])
            feat_after_tri = feat_after_tri_T.T
            if not is_train:
                self.vq[idx_plane].eval()
            else:
                self.vq[idx_plane].train()
            feat_after_tri_vq, _, commit_loss = self.vq[idx_plane](
                feat_after_tri.unsqueeze(0))
            loss += commit_loss.item() / xyz_sampled.shape[0]
            feat_after_tri_vq = feat_after_tri_vq.squeeze(0).T
            plane_coef_point.append(feat_after_tri_vq)
            line_coef_point.append(
                F.grid_sample(
                    self.app_line[idx_plane],
                    coordinate_line[[idx_plane]],
                    align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        # (144,B)          (144,B)
        plane_coef_point, line_coef_point = torch.cat(
            plane_coef_point), torch.cat(line_coef_point)

        return self.basis_mat((plane_coef_point * line_coef_point).T), loss

    def union_prune_and_vq(self, pct_mid, pct_high):
        all_importances = []
        for idx_plane in range(len(self.app_plane)):
            importance = self.importance[f'plane_{idx_plane}']
            all_importances.append(importance.flatten())
        all_importances = torch.cat(all_importances)
        vq_mask, keep_mask = self.cdf_split(
            all_importances, pct_mid, pct_high, prefix='3plane_union')
        plane_mask = []
        for idx_plane in range(len(self.app_plane)):
            importance = self.importance[f'plane_{idx_plane}']
            vq = vq_mask[:importance.numel()]
            kp = keep_mask[:importance.numel()]
            vq_mask = vq_mask[importance.numel():]
            keep_mask = keep_mask[importance.numel():]
            vq = vq.view(importance.shape).squeeze()
            kp = kp.view(importance.shape).squeeze()
            plane_mask.append((vq, kp))
        self.plane_mask = plane_mask

    def split_prune_and_vq(self, pct_mid, pct_high):
        if self.importance is None:
            raise ('Please get importance first')
        plane_mask = []
        for idx_plane in range(len(self.app_plane)):
            importance = self.importance[f'plane_{idx_plane}']
            plane_mask.append(
                self.cdf_split(
                    importance, pct_mid, pct_high,
                    prefix=f'plane_{idx_plane}'))
        self.plane_mask = plane_mask

    def cdf_split(self, importance, pct_mid, pct_high, prefix=''):
        shape = importance.shape
        importance = importance.flatten()
        percent_sum = pct_mid
        vals, idx = sorted_importance = torch.sort(importance + (1e-6))
        cumsum_val = torch.cumsum(vals, dim=0)
        split_index = ((cumsum_val / vals.sum()) >
                       (1 - percent_sum)).nonzero().min()
        split_val_nonprune = vals[split_index]
        percent_point = (importance + (1e-6) >=
                         vals[split_index]).sum() / importance.numel()
        print(
            f'{prefix} {percent_point*100:.4f}% of most important points contribute over {(percent_sum)*100:.4f}%, split_val: {split_val_nonprune}'
        )

        percent_sum = pct_high
        vals, idx = sorted_importance = torch.sort(importance + (1e-6))
        cumsum_val = torch.cumsum(vals, dim=0)
        split_index = ((cumsum_val / vals.sum()) >
                       (1 - percent_sum)).nonzero().min()
        split_val_keep = vals[split_index]
        percent_point = (importance + (1e-6) >=
                         vals[split_index]).sum() / importance.numel()
        print(
            f'{prefix} {percent_point*100:.4f}% of most important points contribute over {(percent_sum)*100:.4f}%, split_val: {split_val_keep} '
        )

        notprune_mask = (importance + (1e-6)) >= split_val_nonprune
        keep_mask = (importance + (1e-6)) >= split_val_keep
        vq_mask = notprune_mask ^ keep_mask
        # vq_mask = torch.flipud(vq_mask)
        # keep_mask = torch.flipud(keep_mask)
        # print("-----------FLIPUD")
        vq_mask = vq_mask.view(shape).squeeze()  #(h*w) -> (1,1,h,w) -> (h,w)
        keep_mask = keep_mask.view(shape).squeeze()
        return (vq_mask, keep_mask)

    def init_vq(self, iteration=100):
        # print("initial vector quantize")
        for _ in tqdm(range(iteration), desc='inital vector quantize'):
            loss_list = []
            for idx_plane in range(len(self.app_plane)):
                vq = self.vq[idx_plane]
                vq.train()
                feat_needvq = self.app_plane[idx_plane].reshape(
                    self.app_n_comp[idx_plane], -1).T
                CHUNK = 2048
                loss = 0
                for i in range(0, feat_needvq.shape[0], CHUNK):
                    ret, indices, commit_loss = vq(
                        feat_needvq[i:i + CHUNK, :].unsqueeze(0))
                    loss += commit_loss
                loss_list.append(loss.item() / feat_needvq.shape[0])
        print(loss_list)

    def train_vq_with_mask(self,
                           iteration=1000,
                           deal_reveal: Union[int, List[int]] = 0,
                           CHUNK=81920):
        for idx_plane in range(len(self.app_plane)):
            vq = self.vq[idx_plane]
            (vq_mask, keep_mask) = self.plane_mask[idx_plane]
            feat_needvq = self.app_plane[idx_plane][:, :, vq_mask].reshape(
                self.app_n_comp[idx_plane], -1).T
            vq.train()
            for i in tqdm(range(iteration)):
                indexes = torch.randint(
                    low=0, high=feat_needvq.shape[0], size=[CHUNK])
                ret, indices, commit = vq(feat_needvq[indexes, :].unsqueeze(0))
                k = deal_reveal if isinstance(deal_reveal,
                                              int) else deal_reveal[1]
                if i < iteration // 2 and k > 0:
                    rand_idx = torch.randint(
                        low=0, high=feat_needvq.shape[0], size=[k])
                    new_code = feat_needvq[rand_idx, :]
                    _, replace_index = torch.topk(
                        vq._codebook.cluster_size, k=k, largest=False)
                    vq._codebook.embed[:, replace_index, :] = new_code

        for idx_plane in range(len(self.density_plane)):
            vq = self.den_vq[idx_plane]
            (vq_mask, keep_mask) = self.plane_mask[idx_plane]
            feat_needvq = self.density_plane[idx_plane][:, :, vq_mask].reshape(
                self.density_n_comp[idx_plane], -1).T
            vq.train()
            for i in tqdm(range(iteration)):
                indexes = torch.randint(
                    low=0, high=feat_needvq.shape[0], size=[CHUNK])
                ret, indices, commit = vq(feat_needvq[indexes, :].unsqueeze(0))
                k = deal_reveal if isinstance(deal_reveal,
                                              int) else deal_reveal[1]
                if i < iteration // 2 and k > 0:
                    rand_idx = torch.randint(
                        low=0, high=feat_needvq.shape[0], size=[k])
                    new_code = feat_needvq[rand_idx, :]
                    _, replace_index = torch.topk(
                        vq._codebook.cluster_size, k=k, largest=False)
                    vq._codebook.embed[:, replace_index, :] = new_code

    def train_vq_with_mask_imp2(self,
                                importance,
                                iteration=1000,
                                deal_reveal: Union[int, List[int]] = 0,
                                CHUNK=81920):
        for idx_plane in range(len(self.app_plane)):
            vq = self.vq[idx_plane]
            (vq_mask, keep_mask) = self.plane_mask[idx_plane]
            feat_needvq = self.app_plane[idx_plane][:, :, vq_mask].reshape(
                self.app_n_comp[idx_plane], -1).T
            vq.train()
            imp_plane = importance[f'plane_{idx_plane}'].reshape(1, -1).T
            print('imp_plane', imp_plane.shape, feat_needvq.shape)
            for i in tqdm(range(iteration)):
                indexes = torch.randint(
                    low=0, high=feat_needvq.shape[0], size=[CHUNK])
                imp = imp_plane[indexes]
                ret, indices, commit = vq(feat_needvq[indexes, :].unsqueeze(0))
                k = deal_reveal if isinstance(deal_reveal,
                                              int) else deal_reveal[1]
                if i < iteration // 2 and k > 0:
                    _, imp_idx = torch.topk(imp.T, k=k)
                    new_code = feat_needvq[imp_idx[0, :], :]
                    _, replace_index = torch.topk(
                        vq._codebook.cluster_size, k=k, largest=False)
                    vq._codebook.embed[:, replace_index, :] = new_code
        for idx_plane in range(len(self.density_plane)):
            vq = self.den_vq[idx_plane]
            (vq_mask, keep_mask) = self.plane_mask[idx_plane]
            feat_needvq = self.density_plane[idx_plane][:, :, vq_mask].reshape(
                self.density_n_comp[idx_plane], -1).T
            vq.train()
            imp_plane = importance[f'plane_{idx_plane}'].reshape(1, -1).T
            for i in tqdm(range(iteration)):
                indexes = torch.randint(
                    low=0, high=feat_needvq.shape[0], size=[CHUNK])
                imp = imp_plane[indexes]
                ret, indices, commit = vq(feat_needvq[indexes, :].unsqueeze(0))
                k = deal_reveal if isinstance(deal_reveal,
                                              int) else deal_reveal[1]
                if i < iteration // 2 and k > 0:
                    _, imp_idx = torch.topk(imp.T, k=k)
                    new_code = feat_needvq[imp_idx[0, :], :]
                    _, replace_index = torch.topk(
                        vq._codebook.cluster_size, k=k, largest=False)
                    vq._codebook.embed[:, replace_index, :] = new_code

    def train_vq_with_mask_imp(self,
                               importance,
                               iteration=1000,
                               deal_reveal: Union[int, List[int]] = 0):
        for idx_plane in range(len(self.app_plane)):
            vq = self.vq[idx_plane]
            (vq_mask, keep_mask) = self.plane_mask[idx_plane]
            feat_needvq = self.app_plane[idx_plane][:, :, vq_mask].reshape(
                self.app_n_comp[idx_plane], -1).T
            vq.train()
            CHUNK = 81920
            imp_plane = importance[f'plane_{idx_plane}'].reshape(1, -1).T
            for i in tqdm(range(iteration)):
                indexes = torch.randint(
                    low=0, high=feat_needvq.shape[0], size=[CHUNK])
                imp = imp_plane[indexes]
                ret, indices, commit = vq(
                    feat_needvq[indexes, :].unsqueeze(0),
                    weight=imp.reshape(1, -1, 1))
                k = deal_reveal if isinstance(deal_reveal,
                                              int) else deal_reveal[1]
                if i < iteration // 2 and k > 0:
                    rand_idx = torch.randint(
                        low=0, high=feat_needvq.shape[0], size=[k])
                    new_code = feat_needvq[rand_idx, :]
                    _, replace_index = torch.topk(
                        vq._codebook.cluster_size, k=k, largest=False)
                    vq._codebook.embed[:, replace_index, :] = new_code

    @torch.no_grad()
    def fully_vq_both(self) -> list:
        all_indices = []
        for idx_plane in tqdm(range(len(self.app_plane))):
            vq = self.vq[idx_plane]
            vq.eval()
            all_feat = self.app_plane[idx_plane].reshape(
                self.app_n_comp[idx_plane], -1).T  # (1,c,h,w)->(h*w,c)
            CHUNK = 2048
            vq_data, indice_list = [], []
            for i in range(0, all_feat.shape[0], CHUNK):
                ret, indices, commit_loss = vq(
                    all_feat[i:i + CHUNK, :].unsqueeze(0))
                vq_data.append(ret[0])
                indice_list.append(indices[0])
            vq.train()
            vq_data = torch.cat(vq_data, dim=0)
            vq_data = vq_data.T.reshape(*(self.app_plane[idx_plane].shape))
            indice = torch.cat(indice_list, dim=0)  # (h*w)
            indice = indice.reshape(
                [1, 1, *self.app_plane[idx_plane].shape[-2:]])

            (vq_mask, keep_mask) = self.plane_mask[idx_plane]

            new_app_plane = torch.zeros_like(self.app_plane[idx_plane])

            new_app_plane[:, :,
                          keep_mask] = self.app_plane[idx_plane][:, :,
                                                                 keep_mask]
            new_app_plane[:, :, vq_mask] = vq_data[:, :, vq_mask]

            self.app_plane[idx_plane].copy_(new_app_plane)
            indice[:, :,
                   keep_mask] = self.vq[idx_plane].codebook_size + torch.arange(
                       keep_mask.sum(), device=self.device)
            active_clustersN = indice.unique().size()[0] - keep_mask.sum()
            # print(f"app_{idx_plane} dead cluster:", self.vq[idx_plane].codebook_size - active_clustersN)
            all_indices.append(indice)
        for idx_plane in tqdm(range(len(self.density_plane))):
            vq = self.den_vq[idx_plane]
            vq.eval()
            all_feat = self.density_plane[idx_plane].reshape(
                self.density_n_comp[idx_plane], -1).T  # (1,c,h,w)->(h*w,c)
            CHUNK = 2048
            vq_data, indice_list = [], []
            for i in range(0, all_feat.shape[0], CHUNK):
                ret, indices, commit_loss = vq(
                    all_feat[i:i + CHUNK, :].unsqueeze(0))
                vq_data.append(ret[0])
                indice_list.append(indices[0])
            vq.train()
            vq_data = torch.cat(vq_data, dim=0)
            vq_data = vq_data.T.reshape(*(self.density_plane[idx_plane].shape))
            indice = torch.cat(indice_list, dim=0)  # (h*w)
            indice = indice.reshape(
                [1, 1, *self.density_plane[idx_plane].shape[-2:]])

            (vq_mask, keep_mask) = self.plane_mask[idx_plane]
            new_density_plane = torch.zeros_like(self.density_plane[idx_plane])

            new_density_plane[:, :, keep_mask] = self.density_plane[
                idx_plane][:, :, keep_mask]
            new_density_plane[:, :, vq_mask] = vq_data[:, :, vq_mask]
            self.density_plane[idx_plane].copy_(new_density_plane)

            indice[:, :, keep_mask] = self.den_vq[
                idx_plane].codebook_size + torch.arange(
                    keep_mask.sum(), device=self.device)
            active_clustersN = indice.unique().size()[0] - keep_mask.sum()
            # print(f"den_{idx_plane} dead cluster:", self.vq[idx_plane].codebook_size - active_clustersN)
            all_indices.append(indice)

        self.all_indices = all_indices
        return all_indices

    @torch.no_grad()
    def saving1(self, savedir, overwrite=False):
        import os, math
        saving_pattern = 'raw'
        dirname = os.path.join(savedir, saving_pattern)
        if overwrite or not os.path.exists(f'{savedir}/{saving_pattern}.zip'):
            print(dirname)
            os.makedirs(dirname, exist_ok=True)
            torch.save(self.density_plane.state_dict(),
                       f'{dirname}/density_plane.pt')
            torch.save(self.density_line.state_dict(),
                       f'{dirname}/density_line.pt')
            torch.save(self.app_plane.state_dict(), f'{dirname}/app_plane.pt')
            torch.save(self.app_line.state_dict(), f'{dirname}/app_line.pt')
            torch.save(self.basis_mat.state_dict(), f'{dirname}/basis_mat.pt')
            torch.save(self.renderModule.state_dict(), f'{dirname}/mlp.pt')

            os.system(f'zip -r {savedir}/{saving_pattern}.zip {dirname} ')
        print(f'saving [{saving_pattern}] size:',
              getsize(f'{savedir}/{saving_pattern}.zip'))

    @torch.no_grad()
    def saving2(self, savedir, overwrite=False):
        import os, math
        saving_pattern = 'raw_half'
        dirname = os.path.join(savedir, saving_pattern)
        if overwrite or not os.path.exists(f'{savedir}/{saving_pattern}.zip'):
            print(dirname)
            os.makedirs(dirname, exist_ok=True)
            save_f = np.savez_compressed

            torch.save(self.density_plane.half().state_dict(),
                       f'{dirname}/density_plane.pt')
            torch.save(self.density_line.half().state_dict(),
                       f'{dirname}/density_line.pt')
            torch.save(self.app_plane.half().state_dict(),
                       f'{dirname}/app_plane.pt')
            torch.save(self.app_line.half().state_dict(),
                       f'{dirname}/app_line.pt')
            torch.save(self.basis_mat.half().state_dict(),
                       f'{dirname}/basis_mat.pt')
            torch.save(self.renderModule.half().state_dict(),
                       f'{dirname}/mlp.pt')
            os.system(f'zip -r {savedir}/{saving_pattern}.zip {dirname} ')
        print(f'saving [{saving_pattern}] size:',
              getsize(f'{savedir}/{saving_pattern}.zip'))

    @torch.no_grad()
    def saving4(self, savedir, overwrite=False):
        import os, math
        saving_pattern = 'vq_both_half'
        dirname = os.path.join(savedir, saving_pattern)
        if overwrite or not os.path.exists(f'{savedir}/{saving_pattern}.zip'):
            print(dirname)
            os.makedirs(dirname, exist_ok=True)
            save_f = np.savez_compressed
            for idx_plane in range(len(self.app_plane)):
                (vq_mask, keep_mask) = self.plane_mask[idx_plane]
                all_indice = self.all_indices[idx_plane]
                save_f(f'{dirname}/vq_mask_{idx_plane}.npz',
                       vq_mask.bool().cpu().numpy())
                save_f(f'{dirname}/keep_mask_{idx_plane}.npz',
                       keep_mask.bool().cpu().numpy())
                save_f(
                    f'{dirname}/codebook_{idx_plane}.npz', self.vq[idx_plane].
                    _codebook.embed.detach().cpu().half().numpy())
                save_f(f'{dirname}/vq_indice_{idx_plane}.npz',
                       all_indice[:, :, vq_mask].cpu().numpy())
                save_f(
                    f'{dirname}/keep_data_{idx_plane}.npz',
                    self.app_plane[idx_plane]
                    [:, :, keep_mask].detach().cpu().half().numpy())
            for idx_plane in range(len(self.density_plane)):
                (vq_mask, keep_mask) = self.plane_mask[idx_plane]
                all_indice = self.all_indices[idx_plane + 3]
                save_f(
                    f'{dirname}/codebook_den_{idx_plane}.npz',
                    self.den_vq[idx_plane]._codebook.embed.detach().cpu().half(
                    ).numpy())
                save_f(f'{dirname}/den_vq_indice_{idx_plane}.npz',
                       all_indice[:, :, vq_mask].cpu().numpy())
                save_f(
                    f'{dirname}/den_data_{idx_plane}.npz',
                    self.density_plane[idx_plane]
                    [:, :, keep_mask].detach().cpu().half().numpy())

            # BUG .half() will change the value of self.app_line, use deepcopy instead
            from copy import deepcopy
            torch.save(
                deepcopy(self.density_line).half().state_dict(),
                f'{dirname}/density_line.pt')
            torch.save(
                deepcopy(self.app_line).half().state_dict(),
                f'{dirname}/app_line.pt')
            torch.save(
                deepcopy(self.basis_mat).half().state_dict(),
                f'{dirname}/basis_mat.pt')
            torch.save(
                deepcopy(self.renderModule).half().state_dict(),
                f'{dirname}/mlp.pt')
            os.system(f'zip -r {savedir}/{saving_pattern}.zip {dirname} ')
        print(f'saving [{saving_pattern}] size:',
              getsize(f'{savedir}/{saving_pattern}.zip'))

    @torch.no_grad()
    def saving5(self, savedir, overwrite=False):
        import os, math
        saving_pattern = 'vq_both_quant'
        dirname = os.path.join(savedir, saving_pattern)
        if overwrite or not os.path.exists(f'{savedir}/{saving_pattern}.zip'):
            print(dirname)
            os.makedirs(dirname, exist_ok=True)
            save_f = np.savez_compressed

            for idx_plane in range(len(self.app_plane)):
                (vq_mask, keep_mask) = self.plane_mask[idx_plane]
                all_indice = self.all_indices[idx_plane]
                save_f(f'{dirname}/vq_mask_{idx_plane}.npz',
                       vq_mask.bool().cpu().numpy())
                save_f(f'{dirname}/keep_mask_{idx_plane}.npz',
                       keep_mask.bool().cpu().numpy())
                save_f(
                    f'{dirname}/codebook_{idx_plane}.npz', self.vq[idx_plane].
                    _codebook.embed.detach().cpu().half().numpy())
                save_f(f'{dirname}/vq_indice_{idx_plane}.npz',
                       all_indice[:, :, vq_mask].cpu().numpy())
                app_data = self.app_plane[idx_plane][:, :, keep_mask].reshape(
                    self.app_n_comp[idx_plane], -1).T
                quant_app_data = torch.quantize_per_channel(
                    app_data,
                    scales=app_data.std(dim=0) / 15,
                    zero_points=app_data.mean(dim=0),
                    axis=1,
                    dtype=torch.qint8)
                save_f(
                    f'{dirname}/quant_keep_data_{idx_plane}.npz', **{
                        f'int_repr':
                        quant_app_data.int_repr().cpu().numpy(),
                        f'scale':
                        quant_app_data.q_per_channel_scales().cpu().numpy(),
                        f'zero_points':
                        quant_app_data.q_per_channel_zero_points().cpu().numpy(
                        ),
                    })
            for idx_plane in range(len(self.density_plane)):
                (vq_mask, keep_mask) = self.plane_mask[idx_plane]
                all_indice = self.all_indices[idx_plane + 3]
                save_f(
                    f'{dirname}/codebook_den_{idx_plane}.npz',
                    self.den_vq[idx_plane]._codebook.embed.detach().cpu().half(
                    ).numpy())
                save_f(f'{dirname}/den_vq_indice_{idx_plane}.npz',
                       all_indice[:, :, vq_mask].cpu().numpy())
                den_data = self.density_plane[
                    idx_plane][:, :, keep_mask].reshape(
                        self.density_n_comp[idx_plane], -1).T
                quant_den_data = torch.quantize_per_channel(
                    den_data,
                    scales=den_data.std(dim=0) / 15,
                    zero_points=den_data.mean(dim=0),
                    axis=1,
                    dtype=torch.qint8)
                save_f(
                    f'{dirname}/quant_den_data_{idx_plane}.npz', **{
                        f'int_repr':
                        quant_den_data.int_repr().cpu().numpy(),
                        f'scale':
                        quant_den_data.q_per_channel_scales().cpu().numpy(),
                        f'zero_points':
                        quant_den_data.q_per_channel_zero_points().cpu().numpy(
                        ),
                    })

            from copy import deepcopy
            torch.save(
                deepcopy(self.density_line).half().state_dict(),
                f'{dirname}/density_line.pt')
            torch.save(
                deepcopy(self.app_line).half().state_dict(),
                f'{dirname}/app_line.pt')
            torch.save(
                deepcopy(self.basis_mat).half().state_dict(),
                f'{dirname}/basis_mat.pt')
            torch.save(
                deepcopy(self.renderModule).half().state_dict(),
                f'{dirname}/mlp.pt')
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            save_f(
                f'{dirname}/alphaMask.npz', **{
                    f'aabb': self.alphaMask.aabb.cpu().numpy(),
                    f'mask': np.packbits(alpha_volume.reshape(-1)),
                    f'shape': alpha_volume.shape,
                })
            os.system(f'zip -r {savedir}/{saving_pattern}.zip {dirname} ')
        print(f'saving [{saving_pattern}] size:',
              getsize(f'{savedir}/{saving_pattern}.zip'))

    @torch.no_grad()
    def extreme_save(self, savedir):
        import math
        kwargs = self.get_kwargs()
        kwargs.update({'codebook_size': self.codebook_size})
        kwargs.update({'use_cosine_sim': self.use_cosine_sim})
        kwargs.update({'codebook_dim': self.codebook_dim})
        ckpt = {'kwargs': kwargs}
        ckpt.update({'density_line': self.density_line.half().state_dict()})
        ckpt.update({'app_line': self.app_line.half().state_dict()})
        ckpt.update({'basis_mat': self.basis_mat.half().state_dict()})
        ckpt.update({'mlp': self.renderModule.half().state_dict()})
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape': alpha_volume.shape})
            ckpt.update(
                {'alphaMask.mask': np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        for idx_plane in range(len(self.app_plane)):
            (vq_mask, keep_mask) = self.plane_mask[idx_plane]
            all_indice = self.all_indices[idx_plane]
            ckpt.update({
                f'vq_mask_{idx_plane}':
                np.packbits(vq_mask.bool().cpu().reshape(-1))
            })
            ckpt.update({
                f'keep_mask_{idx_plane}':
                np.packbits(keep_mask.bool().cpu().reshape(-1))
            })
            ckpt.update({
                f'codebook_{idx_plane}':
                self.vq[idx_plane]._codebook.embed.detach().cpu().half()
            })
            ckpt.update({
                f'vq_indice_{idx_plane}':
                np.packbits(
                    dec2bin(all_indice[:, :, vq_mask].cpu(),
                            int(math.log2(
                                self.codebook_size))).bool().reshape(-1))
            })
            app_data = self.app_plane[idx_plane][:, :, keep_mask].reshape(
                self.app_n_comp[idx_plane], -1).T
            quant_app_data = torch.quantize_per_channel(
                app_data,
                scales=app_data.std(dim=0) / 15,
                zero_points=app_data.mean(dim=0),
                axis=1,
                dtype=torch.qint8)
            ckpt.update({
                f'quant_keep_data_{idx_plane}.int_repr':
                quant_app_data.int_repr().cpu()
            })
            ckpt.update({
                f'quant_keep_data_{idx_plane}.scale':
                quant_app_data.q_per_channel_scales().cpu()
            })
            ckpt.update({
                f'quant_keep_data_{idx_plane}.zero_points':
                quant_app_data.q_per_channel_zero_points().cpu()
            })
        for idx_plane in range(len(self.density_plane)):
            (vq_mask, keep_mask) = self.plane_mask[idx_plane]
            all_indice = self.all_indices[idx_plane + 3]
            ckpt.update({
                f'codebook_den_{idx_plane}':
                self.den_vq[idx_plane]._codebook.embed.detach().cpu().half()
            })
            ckpt.update({
                f'den_vq_indice_{idx_plane}':
                np.packbits(
                    dec2bin(all_indice[:, :, vq_mask].cpu(),
                            int(math.log2(
                                self.codebook_size))).bool().reshape(-1))
            })
            den_data = self.density_plane[idx_plane][:, :, keep_mask].reshape(
                self.density_n_comp[idx_plane], -1).T
            quant_den_data = torch.quantize_per_channel(
                den_data,
                scales=den_data.std(dim=0) / 15,
                zero_points=den_data.mean(dim=0),
                axis=1,
                dtype=torch.qint8)
            ckpt.update({
                f'quant_den_data_{idx_plane}.int_repr':
                quant_den_data.int_repr().cpu()
            })
            ckpt.update({
                f'quant_den_data_{idx_plane}.scale':
                quant_den_data.q_per_channel_scales().cpu()
            })
            ckpt.update({
                f'quant_den_data_{idx_plane}.zero_points':
                quant_den_data.q_per_channel_zero_points().cpu()
            })

        torch.save(ckpt, f'{savedir}/extreme_ckpt.pt')
        print(f'saving [extreme_ckpt] size:',
              getsize(f'{savedir}/extreme_ckpt.pt'))  # 5.31M for lego
        os.system(f'zip {savedir}/extreme_ckpt.zip {savedir}/extreme_ckpt.pt ')
        print(f'saving [extreme_ckpt_npz] size:',
              getsize(f'{savedir}/extreme_ckpt.zip'))  # 3.68M for lego
        return f'{savedir}/extreme_ckpt.pt'

    def extreme_load(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(
                np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(
                    ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(
                self.device, ckpt['alphaMask.aabb'].to(self.device),
                alpha_volume.float().to(self.device))

        # 1. load non-vq part
        self.density_line.load_state_dict(ckpt['density_line'])
        self.app_line.load_state_dict(ckpt['app_line'])
        self.basis_mat.load_state_dict(ckpt['basis_mat'])
        self.renderModule.load_state_dict(ckpt['mlp'])

        # 2. load vq part
        ## load vq_mask, keep_mask
        self.plane_mask = []
        for i in range(3):
            mask_shape = self.app_plane[i].shape[-2:]
            vq_mask = np.unpackbits(
                ckpt[f'vq_mask_{i}'],
                count=np.prod(mask_shape)).reshape(mask_shape).astype(bool)
            keep_mask = np.unpackbits(
                ckpt[f'keep_mask_{i}'],
                count=np.prod(mask_shape)).reshape(mask_shape).astype(bool)
            self.plane_mask.append((vq_mask, keep_mask))

        ## recover app_plane, density_plane
        import math
        bits = int(math.log2(self.codebook_size))
        for idx_plane in range(3):
            (vq_mask, keep_mask) = self.plane_mask[idx_plane]
            # load appearance keep data from quantized data
            int_repr = ckpt[f'quant_keep_data_{idx_plane}.int_repr']
            scale = ckpt[f'quant_keep_data_{idx_plane}.scale']
            zero_points = ckpt[f'quant_keep_data_{idx_plane}.zero_points']
            dequant = (int_repr - zero_points) * scale
            keep_data = dequant.T.reshape(
                *self.app_plane[idx_plane][:, :, keep_mask].shape)
            self.app_plane[idx_plane].data[:, :, keep_mask] = keep_data

            # load appearance vq data from codebook
            codebook = ckpt[f'codebook_{idx_plane}'].float()  #
            vq_count = int(vq_mask.sum())
            unpack1 = np.unpackbits(
                ckpt[f'vq_indice_{idx_plane}'], count=vq_count * bits)
            unpack2 = bin2dec(
                torch.from_numpy(unpack1).reshape(vq_count, bits).long(),
                bits=bits)
            vq_data = codebook[0, unpack2, :]  # N*len
            vq_data = vq_data.T.reshape(
                *(self.app_plane[idx_plane][:, :, vq_mask].shape))
            self.app_plane[idx_plane].data[:, :, vq_mask] = vq_data

        for idx_plane in range(3):
            (vq_mask, keep_mask) = self.plane_mask[idx_plane]
            # load density keep data from quantized data
            int_repr = ckpt[f'quant_den_data_{idx_plane}.int_repr']
            scale = ckpt[f'quant_den_data_{idx_plane}.scale']
            zero_points = ckpt[f'quant_den_data_{idx_plane}.zero_points']
            dequant = (int_repr - zero_points) * scale
            keep_data = dequant.T.reshape(
                *self.density_plane[idx_plane][:, :, keep_mask].shape)
            self.density_plane[idx_plane].data[:, :, keep_mask] = keep_data

            # load density vq data from codebook
            codebook = ckpt[f'codebook_den_{idx_plane}'].float()  #
            vq_count = int(vq_mask.sum())
            unpack1 = np.unpackbits(
                ckpt[f'den_vq_indice_{idx_plane}'], count=vq_count * bits)
            unpack2 = bin2dec(
                torch.from_numpy(unpack1).reshape(vq_count, bits).long(),
                bits=bits)
            vq_data = codebook[0, unpack2, :]  # N*len
            vq_data = vq_data.T.reshape(
                *(self.density_plane[idx_plane][:, :, vq_mask].shape))
            self.density_plane[idx_plane].data[:, :, vq_mask] = vq_data

    @torch.no_grad()
    def quant(self):
        for idx_plane in range(len(self.app_plane)):
            (vq_mask, keep_mask) = self.plane_mask[idx_plane]
            print(keep_mask.sum())
            app_data = self.app_plane[idx_plane][:, :, keep_mask].reshape(
                self.app_n_comp[idx_plane], -1).T
            quant_app_data = torch.quantize_per_channel(
                app_data,
                scales=app_data.std(dim=0) / 15,
                zero_points=app_data.mean(dim=0),
                axis=1,
                dtype=torch.qint8)
            dequant = quant_app_data.dequantize()
            dequant = dequant.T.reshape(
                *self.app_plane[idx_plane][:, :, keep_mask].shape)
            self.app_plane[idx_plane][:, :, keep_mask] = dequant

        for idx_plane in range(len(self.density_plane)):
            (vq_mask, keep_mask) = self.plane_mask[idx_plane]
            density_data = self.density_plane[
                idx_plane][:, :,
                           keep_mask].reshape(self.density_n_comp[idx_plane],
                                              -1).T
            quant_density_data = torch.quantize_per_channel(
                density_data,
                scales=density_data.std(dim=0) / 15,
                zero_points=density_data.mean(dim=0),
                axis=1,
                dtype=torch.qint8)
            dequant = quant_density_data.dequantize()
            dequant = dequant.T.reshape(
                *self.density_plane[idx_plane][:, :, keep_mask].shape)
            self.density_plane[idx_plane][:, :, keep_mask] = dequant

    def forward_train_vq(self,
                         rays_chunk,
                         white_bg=True,
                         is_train=False,
                         ndc_ray=False,
                         N_samples=-1,
                         target=None):
        viewdirs = rays_chunk[:, 3:6]
        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(
                rays_chunk[:, :3],
                viewdirs,
                is_train=is_train,
                N_samples=N_samples)
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(
                    z_vals[:, :1])),
                dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(
                rays_chunk[:, :3],
                viewdirs,
                is_train=is_train,
                N_samples=N_samples)
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(
                    z_vals[:, :1])),
                dim=-1)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3),
                          device=xyz_sampled.device)

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])

            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma

        alpha, weight, bg_weight = raw2alpha(sigma,
                                             dists * self.distance_scale)

        app_mask = weight > self.rayMarch_weight_thres

        vq_loss = None
        if app_mask.any():
            if False:  #isvq and random.random()>0.5:
                app_features, vq_loss = self.compute_appfeature_vq(
                    xyz_sampled[app_mask], is_train)
            else:
                app_features = self.compute_appfeature(xyz_sampled[app_mask])
            valid_rgbs = self.renderModule(xyz_sampled[app_mask],
                                           viewdirs[app_mask], app_features)
            rgb[app_mask] = valid_rgbs

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1, )) < 0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])

        rgb_map = rgb_map.clamp(0, 1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        return rgb_map, depth_map

    def forward(self,
                rays_chunk,
                white_bg=True,
                is_train=False,
                ndc_ray=False,
                N_samples=-1,
                isvq=False):
        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(
                rays_chunk[:, :3],
                viewdirs,
                is_train=is_train,
                N_samples=N_samples)
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(
                    z_vals[:, :1])),
                dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(
                rays_chunk[:, :3],
                viewdirs,
                is_train=is_train,
                N_samples=N_samples)
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(
                    z_vals[:, :1])),
                dim=-1)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3),
                          device=xyz_sampled.device)

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])

            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma

        alpha, weight, bg_weight = raw2alpha(sigma,
                                             dists * self.distance_scale)

        app_mask = weight > self.rayMarch_weight_thres

        vq_loss = None
        if app_mask.any():
            if False:  #isvq and random.random()>0.5:
                app_features, vq_loss = self.compute_appfeature_vq(
                    xyz_sampled[app_mask], is_train)
            else:
                app_features = self.compute_appfeature(xyz_sampled[app_mask])
            valid_rgbs = self.renderModule(xyz_sampled[app_mask],
                                           viewdirs[app_mask], app_features)
            rgb[app_mask] = valid_rgbs

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1, )) < 0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])

        rgb_map = rgb_map.clamp(0, 1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        return rgb_map, depth_map  #, vq_loss # rgb, sigma, alpha, weight, bg_weight

    def save(self, path):
        kwargs = self.get_kwargs()
        kwargs.update({'codebook_size': self.codebook_size})
        kwargs.update({'use_cosine_sim': self.use_cosine_sim})
        kwargs.update({'codebook_dim': self.codebook_dim})
        kwargs.update({
            'importance': self.importance,
            'plane_mask': self.plane_mask,
            'all_indices': self.all_indices
        })
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        # ckpt.update({'vq':{}})
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape': alpha_volume.shape})
            ckpt.update(
                {'alphaMask.mask': np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})

        torch.save(ckpt, path)

    def save_after_vq(self, path):
        raise NotImplementedError

    def load(self, ckpt, load_vq=True):
        if load_vq is False:
            ckpt['state_dict'].update(self.vq.state_dict(prefix='vq.'))
            ckpt['state_dict'].update(self.den_vq.state_dict(prefix='den_vq.'))
        return super().load(ckpt)

    def forward_imp(self,
                    rays_chunk,
                    pseudo_density_planes,
                    is_train=False,
                    ndc_ray=False,
                    N_samples=-1):

        # sample points
        debug = False
        with Timing('-ray preparation', debug):
            viewdirs = rays_chunk[:, 3:6]
            if ndc_ray:
                xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(
                    rays_chunk[:, :3],
                    viewdirs,
                    is_train=is_train,
                    N_samples=N_samples)
                dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1],
                                   torch.zeros_like(z_vals[:, :1])),
                                  dim=-1)
                rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
                dists = dists * rays_norm
                viewdirs = viewdirs / rays_norm
            else:
                xyz_sampled, z_vals, ray_valid = self.sample_ray(
                    rays_chunk[:, :3],
                    viewdirs,
                    is_train=is_train,
                    N_samples=N_samples)
                dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1],
                                   torch.zeros_like(z_vals[:, :1])),
                                  dim=-1)
            viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

        with Timing('-sigma computation', debug):
            if self.alphaMask is not None:
                alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
                alpha_mask = alphas > 0
                ray_invalid = ~ray_valid
                ray_invalid[ray_valid] |= (~alpha_mask)
                ray_valid = ~ray_invalid

            sigma = torch.zeros(
                xyz_sampled.shape[:-1], device=xyz_sampled.device)
            sigma_pseudo = torch.zeros(
                xyz_sampled.shape[:-1], device=xyz_sampled.device)
            rgb = torch.zeros((*xyz_sampled.shape[:2], 3),
                              device=xyz_sampled.device)

            if ray_valid.any():
                xyz_sampled = self.normalize_coord(xyz_sampled)
                sigma_feature = self.compute_densityfeature(
                    xyz_sampled[ray_valid])
                pseudo_sampled = self._compute_pseudo_densityfeature(
                    xyz_sampled[ray_valid], pseudo_density_planes)
                validsigma = self.feature2density(sigma_feature)
                sigma[ray_valid] = validsigma
                sigma_pseudo[ray_valid] = pseudo_sampled
        with Timing('-app render', debug):
            alpha, weight, bg_weight = raw2alpha(sigma,
                                                 dists * self.distance_scale)

        return weight, sigma_pseudo  # rgb, sigma, alpha, weight, bg_weight

    def _compute_pseudo_densityfeature(self,
                                       xyz_sampled,
                                       pseudo_density_planes,
                                       vq_step=False):

        # plane + line basis
        coordinate_plane = torch.stack(
            (xyz_sampled[..., self.matMode[0]], xyz_sampled[...,
                                                            self.matMode[1]],
             xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack(
            (xyz_sampled[..., self.vecMode[0]],
             xyz_sampled[..., self.vecMode[1]], xyz_sampled[...,
                                                            self.vecMode[2]]))
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line),
            dim=-1).detach().view(3, -1, 1, 2)

        sigma_feature = torch.zeros((xyz_sampled.shape[0], ),
                                    device=xyz_sampled.device)
        for idx_plane in range(len(self.density_plane)):

            plane_coef_point = F.grid_sample(
                pseudo_density_planes[idx_plane],
                coordinate_plane[[idx_plane]],
                align_corners=True).view(-1, *xyz_sampled.shape[:1])

            sigma_feature = sigma_feature + plane_coef_point

        return sigma_feature


def getsize(compressed_file, tag='MB'):
    import os
    size = os.path.getsize(compressed_file)
    if tag == 'B':
        pass
    elif tag == 'KB':
        size = size / 1024
    elif tag == 'MB':
        size = size / 1024 / 1024
    elif tag == 'GB':
        size = size / 1024 / 1024 / 1024
    return f'{size} {tag}'
