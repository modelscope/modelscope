# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['VQAutoencoder', 'KLAutoencoder', 'PatchDiscriminator']


def group_norm(dim):
    return nn.GroupNorm(32, dim, eps=1e-6, affine=True)


class Resample(nn.Module):

    def __init__(self, dim, scale_factor):
        super(Resample, self).__init__()
        self.dim = dim
        self.scale_factor = scale_factor

        # layers
        if scale_factor == 2.0:
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                nn.Conv2d(dim, dim, 3, padding=1))
        elif scale_factor == 0.5:
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=2, padding=0))
        else:
            self.resample = nn.Identity()

    def forward(self, x):
        return self.resample(x)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            group_norm(in_dim), nn.SiLU(),
            nn.Conv2d(in_dim, out_dim, 3, padding=1), group_norm(out_dim),
            nn.SiLU(), nn.Dropout(dropout),
            nn.Conv2d(out_dim, out_dim, 3, padding=1))
        self.shortcut = nn.Conv2d(in_dim, out_dim,
                                  1) if in_dim != out_dim else nn.Identity()

        # zero out the last layer params
        nn.init.zeros_(self.residual[-1].weight)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class AttentionBlock(nn.Module):

    def __init__(self, dim):
        super(AttentionBlock, self).__init__()
        self.dim = dim
        self.scale = math.pow(dim, -0.25)

        # layers
        self.norm = group_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()

        # compute query, key, value
        x = self.norm(x)
        q, k, v = self.to_qkv(x).view(b, c * 3, -1).chunk(3, dim=1)

        # compute attention
        attn = torch.einsum('bci,bcj->bij', q * self.scale, k * self.scale)
        attn = F.softmax(attn, dim=-1)

        # gather context
        x = torch.einsum('bij,bcj->bci', attn, v)
        x = x.reshape(b, c, h, w)

        # output
        x = self.proj(x)
        return x + identity


class Encoder(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=3,
                 dim_mult=[1, 2, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 dropout=0.0):
        super(Encoder, self).__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales

        # params
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = nn.Conv2d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                downsamples.append(Resample(out_dim, scale_factor=0.5))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(
            group_norm(out_dim), nn.SiLU(),
            nn.Conv2d(out_dim, z_dim, 3, padding=1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.downsamples(x)
        x = self.middle(x)
        x = self.head(x)
        return x


class Decoder(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=3,
                 dim_mult=[1, 2, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 dropout=0.0):
        super(Decoder, self).__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales

        # params
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = nn.Conv2d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                upsamples.append(Resample(out_dim, scale_factor=2.0))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            group_norm(out_dim), nn.SiLU(),
            nn.Conv2d(out_dim, 3, 3, padding=1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.middle(x)
        x = self.upsamples(x)
        x = self.head(x)
        return x


class VectorQuantizer(nn.Module):

    def __init__(self, codebook_size=8192, z_dim=3, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.codebook_size = codebook_size
        self.z_dim = z_dim
        self.beta = beta

        # init codebook
        eps = math.sqrt(1.0 / codebook_size)
        self.codebook = nn.Parameter(
            torch.empty(codebook_size, z_dim).uniform_(-eps, eps))

    def forward(self, z):
        # preprocess
        b, c, h, w = z.size()
        flatten = z.permute(0, 2, 3, 1).reshape(-1, c)

        # quantization
        with torch.no_grad():
            tokens = torch.cdist(flatten, self.codebook).argmin(dim=1)
        quantized = F.embedding(tokens,
                                self.codebook).view(b, h, w,
                                                    c).permute(0, 3, 1, 2)

        # compute loss
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(quantized.detach(), z)
        loss = codebook_loss + self.beta * commitment_loss

        # perplexity
        counts = F.one_hot(tokens, self.codebook_size).sum(dim=0).to(z.dtype)
        # dist.all_reduce(counts)
        p = counts / counts.sum()
        perplexity = torch.exp(-torch.sum(p * torch.log(p + 1e-10)))

        # postprocess
        tokens = tokens.view(b, h, w)
        quantized = z + (quantized - z).detach()
        return quantized, tokens, loss, perplexity


class VQAutoencoder(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=3,
                 dim_mult=[1, 2, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 dropout=0.0,
                 codebook_size=8192,
                 beta=0.25):
        super(VQAutoencoder, self).__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.codebook_size = codebook_size
        self.beta = beta

        # blocks
        self.encoder = Encoder(dim, z_dim, dim_mult, num_res_blocks,
                               attn_scales, dropout)
        self.conv1 = nn.Conv2d(z_dim, z_dim, 1)
        self.quantizer = VectorQuantizer(codebook_size, z_dim, beta)
        self.conv2 = nn.Conv2d(z_dim, z_dim, 1)
        self.decoder = Decoder(dim, z_dim, dim_mult, num_res_blocks,
                               attn_scales, dropout)

    def forward(self, x):
        z = self.encoder(x)
        z = self.conv1(z)
        z, tokens, loss, perplexity = self.quantizer(z)
        z = self.conv2(z)
        x = self.decoder(z)
        return x, tokens, loss, perplexity

    def encode(self, imgs):
        z = self.encoder(imgs)
        z = self.conv1(z)
        return z

    def decode(self, z):
        r"""Absort the quantizer in the decoder.
        """
        z = self.quantizer(z)[0]
        z = self.conv2(z)
        imgs = self.decoder(z)
        return imgs

    @torch.no_grad()
    def encode_to_tokens(self, imgs):
        # preprocess
        z = self.encoder(imgs)
        z = self.conv1(z)

        # quantization
        b, c, h, w = z.size()
        flatten = z.permute(0, 2, 3, 1).reshape(-1, c)
        tokens = torch.cdist(flatten, self.quantizer.codebook).argmin(dim=1)
        return tokens.view(b, -1)

    @torch.no_grad()
    def decode_from_tokens(self, tokens):
        # dequantization
        z = F.embedding(tokens, self.quantizer.codebook)

        # postprocess
        b, l, c = z.size()
        h = w = int(math.sqrt(l))
        z = z.view(b, h, w, c).permute(0, 3, 1, 2)
        z = self.conv2(z)
        imgs = self.decoder(z)
        return imgs


class KLAutoencoder(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 dropout=0.0):
        super(KLAutoencoder, self).__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales

        # blocks
        self.encoder = Encoder(dim, z_dim * 2, dim_mult, num_res_blocks,
                               attn_scales, dropout)
        self.conv1 = nn.Conv2d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = nn.Conv2d(z_dim, z_dim, 1)
        self.decoder = Decoder(dim, z_dim, dim_mult, num_res_blocks,
                               attn_scales, dropout)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x = self.decode(z)
        return x, mu, log_var

    def encode(self, x):
        x = self.encoder(x)
        mu, log_var = self.conv1(x).chunk(2, dim=1)
        return mu, log_var

    def decode(self, z):
        x = self.conv2(z)
        x = self.decoder(x)
        return x

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu


class PatchDiscriminator(nn.Module):

    def __init__(self, in_dim=3, dim=64, num_layers=3):
        super(PatchDiscriminator, self).__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.num_layers = num_layers

        # params
        dims = [dim * min(8, 2**u) for u in range(num_layers + 1)]

        # layers
        layers = [
            nn.Conv2d(in_dim, dim, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        ]
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            stride = 1 if i == num_layers - 1 else 2
            layers += [
                nn.Conv2d(
                    in_dim, out_dim, 4, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2)
            ]
        layers += [nn.Conv2d(out_dim, 1, 4, stride=1, padding=1)]
        self.layers = nn.Sequential(*layers)

        # initialize weights
        self.apply(self.init_weights)

    def forward(self, x):
        return self.layers(x)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)
