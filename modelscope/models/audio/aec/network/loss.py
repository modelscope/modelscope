# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn.functional as F

from .modulation_loss import (GaborSTRFConv, MelScale,
                              ModulationDomainLossModule)

EPS = 1e-8


def compute_mask(mixed_spec, clean_spec, mask_type='psmiam', clip=1):
    '''
        stft: (batch, ..., 2) or complex(batch, ...)
        y = x + n
    '''
    if torch.is_complex(mixed_spec):
        yr, yi = mixed_spec.real, mixed_spec.imag
    else:
        yr, yi = mixed_spec[..., 0], mixed_spec[..., 1]
    if torch.is_complex(clean_spec):
        xr, xi = clean_spec.real, clean_spec.imag
    else:
        xr, xi = clean_spec[..., 0], clean_spec[..., 1]

    if mask_type == 'iam':
        ymag = torch.sqrt(yr**2 + yi**2)
        xmag = torch.sqrt(xr**2 + xi**2)
        iam = xmag / (ymag + EPS)
        return torch.clamp(iam, 0, 1)

    elif mask_type == 'psm':
        ypow = yr**2 + yi**2
        psm = (xr * yr + xi * yi) / (ypow + EPS)
        return torch.clamp(psm, 0, 1)

    elif mask_type == 'psmiam':
        ypow = yr**2 + yi**2
        psm = (xr * yr + xi * yi) / (ypow + EPS)
        ymag = torch.sqrt(yr**2 + yi**2)
        xmag = torch.sqrt(xr**2 + xi**2)
        iam = xmag / (ymag + EPS)
        psmiam = psm * iam
        return torch.clamp(psmiam, 0, 1)

    elif mask_type == 'crm':
        ypow = yr**2 + yi**2
        mr = (xr * yr + xi * yi) / (ypow + EPS)
        mi = (xi * yr - xr * yi) / (ypow + EPS)
        mr = torch.clamp(mr, -clip, clip)
        mi = torch.clamp(mi, -clip, clip)
        return mr, mi


def energy_vad(spec,
               thdhigh=320 * 600 * 600 * 2,
               thdlow=320 * 300 * 300 * 2,
               int16=True):
    '''
        energy based vad should be accurate enough
        spec: (batch, bins, frames, 2)
        returns (batch, frames)
    '''
    energy = torch.sum(spec[..., 0]**2 + spec[..., 1]**2, dim=1)
    vad = energy > thdhigh
    idx = torch.logical_and(vad == 0, energy > thdlow)
    vad[idx] = 0.5
    return vad


def modulation_loss_init(n_fft):
    gabor_strf_parameters = torch.load(
        './network/gabor_strf_parameters.pt')['state_dict']
    gabor_modulation_kernels = GaborSTRFConv(supn=30, supk=30, nkern=60)
    gabor_modulation_kernels.load_state_dict(gabor_strf_parameters)

    modulation_loss_module = ModulationDomainLossModule(
        gabor_modulation_kernels.eval())
    for param in modulation_loss_module.parameters():
        param.requires_grad = False

    stft2mel = MelScale(
        n_mels=80, sample_rate=16000, n_stft=n_fft // 2 + 1).cuda()

    return modulation_loss_module, stft2mel


def mask_loss_function(
        loss_func='psm_loss',
        loss_type='mse',  # ['mse', 'mae', 'comb']
        mask_type='psmiam',
        use_mod_loss=False,
        use_wav2vec_loss=False,
        n_fft=640,
        hop_length=320,
        EPS=1e-8,
        weight=None):
    if weight is not None:
        print(f'Use loss weight: {weight}')
    winlen = n_fft
    window = torch.hamming_window(winlen, periodic=False)

    def stft(x, return_complex=False):
        # returns [batch, bins, frames, 2]
        return torch.stft(
            x,
            n_fft,
            hop_length,
            winlen,
            window=window.to(x.device),
            center=False,
            return_complex=return_complex)

    def istft(x, slen):
        return torch.istft(
            x,
            n_fft,
            hop_length,
            winlen,
            window=window.to(x.device),
            center=False,
            length=slen)

    def mask_loss(targets, masks, nframes):
        ''' [Batch, Time, Frequency]
        '''
        with torch.no_grad():
            mask_for_loss = torch.ones_like(targets)
            for idx, num in enumerate(nframes):
                mask_for_loss[idx, num:, :] = 0
        masks = masks * mask_for_loss
        targets = targets * mask_for_loss

        if weight is None:
            alpha = 1
        else:  # for aec ST
            alpha = weight - targets

        if loss_type == 'mse':
            loss = 0.5 * torch.sum(alpha * torch.pow(targets - masks, 2))
        elif loss_type == 'mae':
            loss = torch.sum(alpha * torch.abs(targets - masks))
        else:  # mse(mask), mae(mask) approx 1:2
            loss = 0.5 * torch.sum(alpha * torch.pow(targets - masks, 2)
                                   + 0.1 * alpha * torch.abs(targets - masks))
        loss /= torch.sum(nframes)
        return loss

    def spectrum_loss(targets, spec, nframes):
        ''' [Batch, Time, Frequency, 2]
        '''
        with torch.no_grad():
            mask_for_loss = torch.ones_like(targets[..., 0])
            for idx, num in enumerate(nframes):
                mask_for_loss[idx, num:, :] = 0
        xr = spec[..., 0] * mask_for_loss
        xi = spec[..., 1] * mask_for_loss
        yr = targets[..., 0] * mask_for_loss
        yi = targets[..., 1] * mask_for_loss
        xmag = torch.sqrt(spec[..., 0]**2 + spec[..., 1]**2) * mask_for_loss
        ymag = torch.sqrt(targets[..., 0]**2
                          + targets[..., 1]**2) * mask_for_loss

        loss1 = torch.sum(torch.pow(xr - yr, 2) + torch.pow(xi - yi, 2))
        loss2 = torch.sum(torch.pow(xmag - ymag, 2))

        loss = (loss1 + loss2) / torch.sum(nframes)
        return loss

    def sa_loss_dlen(mixed, clean, masks, nframes):
        yspec = stft(mixed).permute([0, 2, 1, 3]) / 32768
        xspec = stft(clean).permute([0, 2, 1, 3]) / 32768
        with torch.no_grad():
            mask_for_loss = torch.ones_like(xspec[..., 0])
            for idx, num in enumerate(nframes):
                mask_for_loss[idx, num:, :] = 0
        emag = ((yspec[..., 0]**2 + yspec[..., 1]**2)**0.15) * (masks**0.3)
        xmag = (xspec[..., 0]**2 + xspec[..., 1]**2)**0.15
        emag = emag * mask_for_loss
        xmag = xmag * mask_for_loss

        loss = torch.sum(torch.pow(emag - xmag, 2)) / torch.sum(nframes)
        return loss

    def psm_vad_loss_dlen(mixed, clean, masks, nframes, subtask=None):
        mixed_spec = stft(mixed)
        clean_spec = stft(clean)
        targets = compute_mask(mixed_spec, clean_spec, mask_type)
        # [B, T, F]
        targets = targets.permute(0, 2, 1)

        loss = mask_loss(targets, masks, nframes)

        if subtask is not None:
            vadtargets = energy_vad(clean_spec)
            with torch.no_grad():
                mask_for_loss = torch.ones_like(targets[:, :, 0])
                for idx, num in enumerate(nframes):
                    mask_for_loss[idx, num:] = 0
            subtask = subtask[:, :, 0] * mask_for_loss
            vadtargets = vadtargets * mask_for_loss

            loss_vad = F.binary_cross_entropy(subtask, vadtargets)
            return loss + loss_vad
        return loss

    def modulation_loss(mixed, clean, masks, nframes, subtask=None):
        mixed_spec = stft(mixed, True)
        clean_spec = stft(clean, True)
        enhanced_mag = torch.abs(mixed_spec)
        clean_mag = torch.abs(clean_spec)
        with torch.no_grad():
            mask_for_loss = torch.ones_like(clean_mag)
            for idx, num in enumerate(nframes):
                mask_for_loss[idx, :, num:] = 0
        clean_mag = clean_mag * mask_for_loss
        enhanced_mag = enhanced_mag * mask_for_loss * masks.permute([0, 2, 1])

        # Covert to log-mel representation
        # (B,T,#mel_channels)
        clean_log_mel = torch.log(
            torch.transpose(stft2mel(clean_mag**2), 2, 1) + 1e-8)
        enhanced_log_mel = torch.log(
            torch.transpose(stft2mel(enhanced_mag**2), 2, 1) + 1e-8)

        alpha = compute_mask(mixed_spec, clean_spec, mask_type)
        alpha = alpha.permute(0, 2, 1)
        loss = 0.05 * modulation_loss_module(enhanced_log_mel, clean_log_mel,
                                             alpha)
        loss2 = psm_vad_loss_dlen(mixed, clean, masks, nframes, subtask)
        # print(loss.item(), loss2.item()) #approx 1:4
        loss = loss + loss2
        return loss

    def wav2vec_loss(mixed, clean, masks, nframes, subtask=None):
        mixed /= 32768
        clean /= 32768
        mixed_spec = stft(mixed)
        with torch.no_grad():
            mask_for_loss = torch.ones_like(masks)
            for idx, num in enumerate(nframes):
                mask_for_loss[idx, num:, :] = 0
        masks_est = masks * mask_for_loss

        estimate = mixed_spec * masks_est.permute([0, 2, 1]).unsqueeze(3)
        est_clean = istft(estimate, clean.shape[1])
        loss = wav2vec_loss_module(est_clean, clean)
        return loss

    def sisdr_loss_dlen(mixed,
                        clean,
                        masks,
                        nframes,
                        subtask=None,
                        zero_mean=True):
        mixed_spec = stft(mixed)
        with torch.no_grad():
            mask_for_loss = torch.ones_like(masks)
            for idx, num in enumerate(nframes):
                mask_for_loss[idx, num:, :] = 0
        masks_est = masks * mask_for_loss

        estimate = mixed_spec * masks_est.permute([0, 2, 1]).unsqueeze(3)
        est_clean = istft(estimate, clean.shape[1])
        flen = min(clean.shape[1], est_clean.shape[1])
        clean = clean[:, :flen]
        est_clean = est_clean[:, :flen]

        # follow asteroid/losses/sdr.py
        if zero_mean:
            clean = clean - torch.mean(clean, dim=1, keepdim=True)
            est_clean = est_clean - torch.mean(est_clean, dim=1, keepdim=True)

        dot = torch.sum(est_clean * clean, dim=1, keepdim=True)
        s_clean_energy = torch.sum(clean**2, dim=1, keepdim=True) + EPS
        scaled_clean = dot * clean / s_clean_energy
        e_noise = est_clean - scaled_clean

        # [batch]
        sisdr = torch.sum(
            scaled_clean**2, dim=1) / (
                torch.sum(e_noise**2, dim=1) + EPS)
        sisdr = -10 * torch.log10(sisdr + EPS)
        loss = sisdr.mean()
        return loss

    def sisdr_freq_loss_dlen(mixed, clean, masks, nframes, subtask=None):
        mixed_spec = stft(mixed)
        clean_spec = stft(clean)
        with torch.no_grad():
            mask_for_loss = torch.ones_like(masks)
            for idx, num in enumerate(nframes):
                mask_for_loss[idx, num:, :] = 0
        masks_est = masks * mask_for_loss

        estimate = mixed_spec * masks_est.permute([0, 2, 1]).unsqueeze(3)

        dot_real = estimate[..., 0] * clean_spec[..., 0] + \
            estimate[..., 1] * clean_spec[..., 1]
        dot_imag = estimate[..., 0] * clean_spec[..., 1] - \
            estimate[..., 1] * clean_spec[..., 0]
        dot = torch.cat([dot_real.unsqueeze(3), dot_imag.unsqueeze(3)], dim=-1)
        s_clean_energy = clean_spec[..., 0] ** 2 + \
            clean_spec[..., 1] ** 2 + EPS
        scaled_clean = dot * clean_spec / s_clean_energy.unsqueeze(3)
        e_noise = estimate - scaled_clean

        # [batch]
        scaled_clean_energy = torch.sum(
            scaled_clean[..., 0]**2 + scaled_clean[..., 1]**2, dim=1)
        e_noise_energy = torch.sum(
            e_noise[..., 0]**2 + e_noise[..., 1]**2, dim=1)
        sisdr = torch.sum(
            scaled_clean_energy, dim=1) / (
                torch.sum(e_noise_energy, dim=1) + EPS)
        sisdr = -10 * torch.log10(sisdr + EPS)
        loss = sisdr.mean()
        return loss

    def crm_loss_dlen(mixed, clean, masks, nframes, subtask=None):
        mixed_spec = stft(mixed).permute([0, 2, 1, 3])
        clean_spec = stft(clean).permute([0, 2, 1, 3])
        mixed_spec = mixed_spec / 32768
        clean_spec = clean_spec / 32768
        tgt_mr, tgt_mi = compute_mask(mixed_spec, clean_spec, mask_type='crm')

        D = int(masks.shape[2] / 2)
        with torch.no_grad():
            mask_for_loss = torch.ones_like(clean_spec[..., 0])
            for idx, num in enumerate(nframes):
                mask_for_loss[idx, num:, :] = 0
        mr = masks[..., :D] * mask_for_loss
        mi = masks[..., D:] * mask_for_loss
        tgt_mr = tgt_mr * mask_for_loss
        tgt_mi = tgt_mi * mask_for_loss

        if weight is None:
            alpha = 1
        else:
            alpha = weight - tgt_mr
        # signal approximation
        yr = mixed_spec[..., 0]
        yi = mixed_spec[..., 1]
        loss1 = torch.sum(alpha * torch.pow((mr * yr - mi * yi) - clean_spec[..., 0], 2)) \
            + torch.sum(alpha * torch.pow((mr * yi + mi * yr) - clean_spec[..., 1], 2))
        # mask approximation
        loss2 = torch.sum(alpha * torch.pow(mr - tgt_mr, 2)) \
            + torch.sum(alpha * torch.pow(mi - tgt_mi, 2))
        loss = 0.5 * (loss1 + loss2) / torch.sum(nframes)
        return loss

    def crm_miso_loss_dlen(mixed, clean, masks, nframes):
        return crm_loss_dlen(mixed[..., 0], clean[..., 0], masks, nframes)

    def mimo_loss_dlen(mixed, clean, masks, nframes):
        chs = mixed.shape[-1]
        D = masks.shape[2] // chs
        loss = psm_vad_loss_dlen(mixed[..., 0], clean[..., 0], masks[..., :D],
                                 nframes)
        for ch in range(1, chs):
            loss1 = psm_vad_loss_dlen(mixed[..., ch], clean[..., ch],
                                      masks[..., ch * D:ch * D + D], nframes)
            loss = loss + loss1
        return loss / chs

    def spec_loss_dlen(mixed, clean, spec, nframes):
        clean_spec = stft(clean).permute([0, 2, 1, 3])
        clean_spec = clean_spec / 32768

        D = spec.shape[2] // 2
        spec_est = torch.cat([spec[..., :D, None], spec[..., D:, None]],
                             dim=-1)
        loss = spectrum_loss(clean_spec, spec_est, nframes)
        return loss

    if loss_func == 'psm_vad_loss_dlen':
        return psm_vad_loss_dlen
    elif loss_func == 'sisdr_loss_dlen':
        return sisdr_loss_dlen
    elif loss_func == 'sisdr_freq_loss_dlen':
        return sisdr_freq_loss_dlen
    elif loss_func == 'crm_loss_dlen':
        return crm_loss_dlen
    elif loss_func == 'modulation_loss':
        return modulation_loss
    elif loss_func == 'wav2vec_loss':
        return wav2vec_loss
    elif loss_func == 'mimo_loss_dlen':
        return mimo_loss_dlen
    elif loss_func == 'spec_loss_dlen':
        return spec_loss_dlen
    elif loss_func == 'sa_loss_dlen':
        return sa_loss_dlen
    else:
        print('error loss func')
        return None
