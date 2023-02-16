# Copyright (c) Alibaba, Inc. and its affiliates.
import torch


def warp(im, flow):

    def _repeat(x, n_repeats):
        rep = torch.ones((1, n_repeats), dtype=torch.int32)
        x = torch.matmul(x.view(-1, 1).int(), rep)
        return x.view(-1)

    def _repeat2(x, n_repeats):
        rep = torch.ones((n_repeats, 1), dtype=torch.int32)
        x = torch.matmul(rep, x.view(1, -1).int())
        return x.view(-1)

    def _interpolate(im, x, y):
        num_batch, channels, height, width = im.shape

        x = x.float()
        y = y.float()
        max_y = height - 1
        max_x = width - 1

        x = _repeat2(torch.arange(0, width),
                     height * num_batch).float().cuda() + x * 64
        y = _repeat2(_repeat(torch.arange(0, height), width),
                     num_batch).float().cuda() + y * 64

        # do sampling
        x0 = (torch.floor(x.cpu())).int()
        x1 = x0 + 1
        y0 = (torch.floor(y.cpu())).int()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)
        dim2 = width
        dim1 = width * height
        base = _repeat(torch.arange(num_batch) * dim1, height * width)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        im_flat = im.permute(0, 2, 3, 1)
        im_flat = im_flat.reshape((-1, channels)).float()
        Ia = torch.gather(
            im_flat, dim=0, index=torch.unsqueeze(idx_a, 1).long().cuda())
        Ib = torch.gather(im_flat, 0, torch.unsqueeze(idx_b, 1).long().cuda())
        Ic = torch.gather(im_flat, 0, torch.unsqueeze(idx_c, 1).long().cuda())
        Id = torch.gather(im_flat, 0, torch.unsqueeze(idx_d, 1).long().cuda())
        # and finally calculate interpolated values
        x0_f = x0.float().cuda()
        x1_f = x1.float().cuda()
        y0_f = y0.float().cuda()
        y1_f = y1.float().cuda()
        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id
        return output

    def _meshgrid(height, width):
        x_t = torch.matmul(
            torch.ones((height, 1)),
            torch.unsqueeze(torch.linspace(-0.1, 0.1, width),
                            1).permute(1, 0)).cuda()
        y_t = torch.matmul(
            torch.unsqueeze(torch.linspace(-0.1, 0.1, height), 1),
            torch.ones((1, width))).cuda()

        x_t_flat = x_t.reshape((1, -1))
        y_t_flat = y_t.reshape((1, -1))

        ones = torch.ones_like(x_t_flat).cuda()
        grid = torch.cat((x_t_flat, y_t_flat, ones), 0)

        return grid

    def _warp(x_s, y_s, input_dim):
        num_batch, num_channels, height, width = input_dim.shape
        # out_height, out_width = out_size

        x_s_flat = x_s.reshape(-1)
        y_s_flat = y_s.reshape(-1)

        input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat)
        output = input_transformed.reshape(
            (num_batch, num_channels, height, width))
        return output

    n_dims = int(flow.shape[1]) // 2
    dx = flow[:, :n_dims, :, :]
    dy = flow[:, n_dims:, :, :]

    output = torch.cat([
        _warp(dx[:, idx:idx + 1, :, :], dy[:, idx:idx + 1, :, :],
              im[:, idx:idx + 1, :, :]) for idx in range(im.shape[1])
    ], 1)
    return output
