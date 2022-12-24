# Adopted from https://github.com/Limingxing00/RDE-VOS-CVPR2022
# under MIT License

import math

import torch


def softmax_w_top(x, top):
    values, indices = torch.topk(x, k=top, dim=1)
    x_exp = values.exp_()

    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    x.zero_().scatter_(1, indices, x_exp)  # B * THW * HW

    return x


def make_gaussian(y_idx, x_idx, height, width, sigma=7):
    yv, xv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])

    yv = yv.reshape(height * width).unsqueeze(0).float().cuda()
    xv = xv.reshape(height * width).unsqueeze(0).float().cuda()

    y_idx = y_idx.transpose(0, 1)
    x_idx = x_idx.transpose(0, 1)

    g = torch.exp(-((yv - y_idx)**2 + (xv - x_idx)**2) / (2 * sigma**2))

    return g


def kmn(x, top=None, gauss=None):
    if top is not None:
        if gauss is not None:
            maxes = torch.max(x, dim=1, keepdim=True)[0]
            x_exp = torch.exp(x - maxes) * gauss
            x_exp, indices = torch.topk(x_exp, k=top, dim=1)
        else:
            values, indices = torch.topk(x, k=top, dim=1)
            x_exp = torch.exp(values - values[:, 0])

        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        x_exp /= x_exp_sum
        x.zero_().scatter_(1, indices, x_exp)  # B * THW * HW

        output = x
    else:
        maxes = torch.max(x, dim=1, keepdim=True)[0]
        if gauss is not None:
            x_exp = torch.exp(x - maxes) * gauss

        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        x_exp /= x_exp_sum
        output = x_exp

    return output


class MemoryBank:

    def __init__(self, compress, k, top_k=20, mode='stm'):
        self.top_k = top_k

        self.CK = None
        self.CV = None

        self.mem_k = None
        self.mem_v = None

        self.num_objects = k
        self.km = 5.6

        self.compress = compress

        self.init_mode(mode)

    def init_mode(self, mode):
        """
        stm, two-frames, gt, last, compress, gt-compress,
        last-compress, two-frames-compress
        """
        self.is_compress = None
        self.use_gt = None
        self.use_last = None
        self.stm = None
        print('mode is {}'.format(mode))
        if mode == 'stm':
            self.stm = True
        elif mode == 'two-frames':
            self.use_gt = True
            self.use_last = True
        elif mode == 'gt':
            self.use_gt = True
        elif mode == 'last':
            self.use_last = True
        elif mode == 'compress':
            self.is_compress = True
        elif mode == 'gt-compress':
            self.use_gt = True
            self.is_compress = True
        elif mode == 'last-compress':
            self.use_last = True
            self.is_compress = True
        elif mode == 'two-frames-compress':
            self.use_gt = True
            self.use_last = True
            self.is_compress = True
        else:
            raise RuntimeError('check mode!')

    def _global_matching(self, mk, qk, H, W):
        # NE means number of elements -- typically T*H*W
        mk = mk.flatten(start_dim=2)
        qk = qk.flatten(start_dim=2)
        B, CK, NE = mk.shape

        a = mk.pow(2).sum(1).unsqueeze(2)
        b = 2 * (mk.transpose(1, 2) @ qk)
        # We don't actually need this, will update paper later
        # c = qk.pow(2).expand(B, -1, -1).sum(1).unsqueeze(1)

        affinity = (-a + b) / math.sqrt(CK)  # B, NE, HW
        # if self.km is not None:
        #     # Make a bunch of Gaussian distributions
        #     argmax_idx = affinity.max(2)[1]
        #     y_idx, x_idx = argmax_idx//W, argmax_idx%W
        #     g = make_gaussian(y_idx, x_idx, H, W, sigma=self.km)
        #     g = g.view(B, NE, H*W)

        #     affinity = kmn(affinity, top=20, gauss=g)  # B, THW, HW
        affinity = softmax_w_top(affinity, top=self.top_k)  # B, THW, HW

        return affinity

    def _readout(self, affinity, mv):
        return torch.bmm(mv, affinity)

    def match_memory(self, qk):
        k = self.num_objects
        _, _, h, w = qk.shape

        qk = qk.flatten(start_dim=2)

        # use gt+last+mem
        if self.temp_k is not None and self.is_compress and self.use_last and \
                self.use_gt:
            # print("mode: gt+last+mem")
            mk = torch.cat([self.mem_k, self.temp_k, self.gt_k, self.gt_k], 2)
            # mv = torch.cat([self.mem_v, self.temp_v], 2)
            try:
                mv = torch.cat([self.mem_v, self.temp_v, self.gt_v, self.gt_v],
                               2)
            except Exception:
                mv = torch.cat([
                    self.mem_v,
                    self.temp_v.unsqueeze(0),
                    self.gt_v.unsqueeze(0),
                    self.gt_v.unsqueeze(0)
                ], 3)
        # use gt+last
        elif self.temp_k is not None and self.use_last and self.use_gt:
            # print("mode: gt+last")
            mk = torch.cat([self.temp_k, self.gt_k, self.gt_k], 2)
            # mv = torch.cat([self.mem_v, self.temp_v], 2)
            try:
                mv = torch.cat([self.temp_v, self.gt_v, self.gt_v], 2)
            except Exception:
                mv = torch.cat([
                    self.temp_v.unsqueeze(0),
                    self.gt_v.unsqueeze(0),
                    self.gt_v.unsqueeze(0)
                ], 3)
        # use last+mem
        elif self.temp_k is not None and self.is_compress and self.use_last:
            # print("mode: last+mem")
            mk = torch.cat([self.mem_k, self.temp_k], 2)
            try:
                mv = torch.cat([self.mem_v, self.temp_v], 2)
            except Exception:
                mv = torch.cat([self.mem_v, self.temp_v.unsqueeze(0)], 3)
        # use gt+mem
        elif self.is_compress and self.use_gt:
            # print("mode: gt+mem")
            # mk = self.mem_k
            # mv = self.mem_v
            mk = torch.cat([self.mem_k, self.gt_k], 2)
            try:
                mv = torch.cat([self.mem_v, self.gt_v], 2)
            except Exception:
                mv = torch.cat([self.mem_v, self.gt_v.unsqueeze(0)], 3)
        # use last
        elif self.temp_k is not None and self.use_last:
            # print("mode: last")
            mk = self.temp_k
            mv = self.temp_v
        # use gt or only use our embedding
        else:
            # print("mode: gt")
            # use nothing
            mk = self.mem_k
            mv = self.mem_v

        affinity = self._global_matching(mk, qk, h, w)
        if len(mv.shape) == 6:
            mv = mv.squeeze(0)
        mv = mv.flatten(start_dim=2)

        # One affinity for all
        readout_mem = self._readout(affinity.expand(k, -1, -1), mv)

        return readout_mem.view(k, self.CV, h, w)

    def add_memory(self, key, value, is_temp=False):
        # Temp is for "last frame"
        # Not always used
        # But can always be flushed
        self.temp_k = None
        self.temp_v = None

        if self.mem_k is None:
            # First frame, just shove it in
            self.mem_k = key  # gt
            self.mem_v = value
            self.CK = key.shape[1]
            self.CV = value.shape[1]
            self.gt_k = key
            self.gt_v = value

        elif self.is_compress:
            # compress the two frames
            if len(self.mem_v.shape) == 5:
                self.mem_v = self.mem_v.unsqueeze(0)
            k = torch.cat([self.mem_k, key], 2)  # [1, 64, 2, 30, 57]
            v = torch.cat([self.mem_v, value.unsqueeze(0)],
                          3)  # [1, 2, 512, 2, 30, 57]
            self.mem_k, self.mem_v = self.compress(k, v)

            # check if use last frame
            if self.use_last:
                self.temp_k = key  # [1, 64, 1, 30, 57]
                self.temp_v = value  # [2, 512, 1, 30, 57]

        elif self.stm:
            # stm style
            # print("stm", self.mem_k.shape)
            self.mem_k = torch.cat([self.mem_k, key], 2)
            self.mem_v = torch.cat([self.mem_v, value], 2)

        else:
            # no compress
            # check if use last frame
            if self.use_last:
                self.temp_k = key
                self.temp_v = value
