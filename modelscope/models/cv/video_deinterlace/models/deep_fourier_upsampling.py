# The implementation is adopted from Deep Fourier Upsampling,
# made publicly available at https://github.com/manman1995/Deep-Fourier-Upsampling
import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F


class freup_Periodicpadding(nn.Module):

    def __init__(self, channels):
        super(freup_Periodicpadding, self).__init__()

        self.amp_fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(channels, channels, 1, 1, 0))

        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):

        N, C, H, W = x.shape

        fft_x = torch.fft.fft(torch.fft.fft(x, dim=0), dim=1)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x).detach()

        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)

        amp_fuse = Mag.repeat(1, 1, 2, 2)
        pha_fuse = Pha.repeat(1, 1, 2, 2)

        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)

        output = torch.fft.ifft(torch.fft.ifft(out, dim=0), dim=1)
        output = torch.abs(output)

        return self.post(output)
