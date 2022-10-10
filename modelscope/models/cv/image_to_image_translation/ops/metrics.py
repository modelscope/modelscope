# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import numpy as np
import scipy.linalg as linalg
import torch

__all__ = [
    'get_fid_net', 'get_is_net', 'compute_fid', 'compute_prdc', 'compute_is'
]


def get_fid_net(resize_input=True, normalize_input=True):
    r"""InceptionV3 network for the evaluation of Fréchet Inception Distance (FID).

    Args:
        resize_input: whether or not to resize the input to (299, 299).
        normalize_input: whether or not to normalize the input from range (0, 1) to range(-1, 1).
    """
    from artist.models import InceptionV3
    return InceptionV3(
        output_blocks=(3, ),
        resize_input=resize_input,
        normalize_input=normalize_input,
        requires_grad=False,
        use_fid_inception=True).eval().requires_grad_(False)


def get_is_net(resize_input=True, normalize_input=True):
    r"""InceptionV3 network for the evaluation of Inception Score (IS).

    Args:
        resize_input: whether or not to resize the input to (299, 299).
        normalize_input: whether or not to normalize the input from range (0, 1) to range(-1, 1).
    """
    from artist.models import InceptionV3
    return InceptionV3(
        output_blocks=(4, ),
        resize_input=resize_input,
        normalize_input=normalize_input,
        requires_grad=False,
        use_fid_inception=False).eval().requires_grad_(False)


@torch.no_grad()
def compute_fid(real_feats, fake_feats, eps=1e-6):
    r"""Compute Fréchet Inception Distance (FID).

    Args:
        real_feats: [N, C].
        fake_feats: [N, C].
    """
    # check inputs
    if isinstance(real_feats, torch.Tensor):
        real_feats = real_feats.cpu().numpy().astype(np.float_)
    if isinstance(fake_feats, torch.Tensor):
        fake_feats = fake_feats.cpu().numpy().astype(np.float_)

    # real statistics
    mu1 = np.mean(real_feats, axis=0)
    sigma1 = np.cov(real_feats, rowvar=False)

    # fake statistics
    mu2 = np.mean(fake_feats, axis=0)
    sigma2 = np.cov(fake_feats, rowvar=False)

    # compute covmean
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(
            f'FID calculation produces singular product; adding {eps} to diagonal of cov',
            flush=True)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    # compute Fréchet distance
    diff = mu1 - mu2
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(
        sigma2) - 2 * np.trace(covmean)
    return fid.item()


@torch.no_grad()
def compute_prdc(real_feats, fake_feats, knn=5):
    r"""Compute precision, recall, density, and coverage given two manifolds.

    Args:
        real_feats: [N, C].
        fake_feats: [N, C].
        knn: the number of nearest neighbors to consider.
    """
    # distances
    real_kth = -(-torch.cdist(real_feats, real_feats)).topk(
        k=knn, dim=1)[0][:, -1]
    fake_kth = -(-torch.cdist(fake_feats, fake_feats)).topk(
        k=knn, dim=1)[0][:, -1]
    dists = torch.cdist(real_feats, fake_feats)

    # metrics
    precision = (dists < real_kth.unsqueeze(1)).any(
        dim=0).float().mean().item()
    recall = (dists < fake_kth.unsqueeze(0)).any(dim=1).float().mean().item()
    density = (dists < real_kth.unsqueeze(1)).float().sum(
        dim=0).mean().item() / knn
    coverage = (dists.min(dim=1)[0] < real_kth).float().mean().item()
    return precision, recall, density, coverage


@torch.no_grad()
def compute_is(logits, num_splits=10):
    preds = logits.softmax(dim=1).cpu().numpy()
    split_scores = []
    for k in range(num_splits):
        part = preds[k * (len(logits) // num_splits):(k + 1)
                     * (len(logits) // num_splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    return np.mean(split_scores), np.std(split_scores)
