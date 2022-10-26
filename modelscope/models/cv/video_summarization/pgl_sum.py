# Part of the implementation is borrowed and modified from PGL-SUM,
# publicly available at https://github.com/e-apostolidis/PGL-SUM

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):

    def __init__(self,
                 input_size=1024,
                 output_size=1024,
                 freq=10000,
                 heads=1,
                 pos_enc=None):
        """ The basic (multi-head) Attention 'cell' containing the learnable parameters of Q, K and V

        :param int input_size: Feature input size of Q, K, V.
        :param int output_size: Feature -hidden- size of Q, K, V.
        :param int freq: The frequency of the sinusoidal positional encoding.
        :param int heads: Number of heads for the attention module.
        :param str | None pos_enc: The type of the positional encoding [supported: Absolute, Relative].
        """
        super(SelfAttention, self).__init__()

        self.permitted_encodings = ['absolute', 'relative']
        if pos_enc is not None:
            pos_enc = pos_enc.lower()
            assert pos_enc in self.permitted_encodings, f'Supported encodings: {*self.permitted_encodings,}'

        self.input_size = input_size
        self.output_size = output_size
        self.heads = heads
        self.pos_enc = pos_enc
        self.freq = freq
        self.Wk, self.Wq, self.Wv = nn.ModuleList(), nn.ModuleList(
        ), nn.ModuleList()
        for _ in range(self.heads):
            self.Wk.append(
                nn.Linear(
                    in_features=input_size,
                    out_features=output_size // heads,
                    bias=False))
            self.Wq.append(
                nn.Linear(
                    in_features=input_size,
                    out_features=output_size // heads,
                    bias=False))
            self.Wv.append(
                nn.Linear(
                    in_features=input_size,
                    out_features=output_size // heads,
                    bias=False))
        self.out = nn.Linear(
            in_features=output_size, out_features=input_size, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(p=0.5)

    def getAbsolutePosition(self, T):
        """Calculate the sinusoidal positional encoding based on the absolute position of each considered frame.
        Based on 'Attention is all you need' paper (https://arxiv.org/abs/1706.03762)

        :param int T: Number of frames contained in Q, K and V
        :return: Tensor with shape [T, T]
        """
        freq = self.freq
        d = self.input_size

        pos = torch.tensor([k for k in range(T)],
                           device=self.out.weight.device)
        i = torch.tensor([k for k in range(T // 2)],
                         device=self.out.weight.device)

        # Reshape tensors each pos_k for each i indices
        pos = pos.reshape(pos.shape[0], 1)
        pos = pos.repeat_interleave(i.shape[0], dim=1)
        i = i.repeat(pos.shape[0], 1)

        AP = torch.zeros(T, T, device=self.out.weight.device)
        AP[pos, 2 * i] = torch.sin(pos / freq**((2 * i) / d))
        AP[pos, 2 * i + 1] = torch.cos(pos / freq**((2 * i) / d))
        return AP

    def getRelativePosition(self, T):
        """Calculate the sinusoidal positional encoding based on the relative position of each considered frame.
        r_pos calculations as here: https://theaisummer.com/positional-embeddings/

        :param int T: Number of frames contained in Q, K and V
        :return: Tensor with shape [T, T]
        """
        freq = self.freq
        d = 2 * T
        min_rpos = -(T - 1)

        i = torch.tensor([k for k in range(T)], device=self.out.weight.device)
        j = torch.tensor([k for k in range(T)], device=self.out.weight.device)

        # Reshape tensors each i for each j indices
        i = i.reshape(i.shape[0], 1)
        i = i.repeat_interleave(i.shape[0], dim=1)
        j = j.repeat(i.shape[0], 1)

        # Calculate the relative positions
        r_pos = j - i - min_rpos

        RP = torch.zeros(T, T, device=self.out.weight.device)
        idx = torch.tensor([k for k in range(T // 2)],
                           device=self.out.weight.device)
        RP[:, 2 * idx] = torch.sin(
            r_pos[:, 2 * idx] / freq**((i[:, 2 * idx] + j[:, 2 * idx]) / d))
        RP[:, 2 * idx + 1] = torch.cos(
            r_pos[:, 2 * idx + 1]
            / freq**((i[:, 2 * idx + 1] + j[:, 2 * idx + 1]) / d))
        return RP

    def forward(self, x):
        """ Compute the weighted frame features, based on either the global or local (multi-head) attention mechanism.

        :param torch.tensor x: Frame features with shape [T, input_size]
        :return: A tuple of:
                    y: Weighted features based on the attention weights, with shape [T, input_size]
                    att_weights : The attention weights (before dropout), with shape [T, T]
        """
        outputs = []
        for head in range(self.heads):
            K = self.Wk[head](x)
            Q = self.Wq[head](x)
            V = self.Wv[head](x)

            # Q *= 0.06                       # scale factor VASNet
            # Q /= np.sqrt(self.output_size)  # scale factor (i.e 1 / sqrt(d_k) )
            energies = torch.matmul(Q, K.transpose(1, 0))
            if self.pos_enc is not None:
                if self.pos_enc == 'absolute':
                    AP = self.getAbsolutePosition(T=energies.shape[0])
                    energies = energies + AP
                elif self.pos_enc == 'relative':
                    RP = self.getRelativePosition(T=energies.shape[0])
                    energies = energies + RP

            att_weights = self.softmax(energies)
            _att_weights = self.drop(att_weights)
            y = torch.matmul(_att_weights, V)

            # Save the current head output
            outputs.append(y)
        y = self.out(torch.cat(outputs, dim=1))
        return y, att_weights.clone(
        )  # for now we don't deal with the weights (probably max or avg pooling)


class MultiAttention(nn.Module):

    def __init__(self,
                 input_size=1024,
                 output_size=1024,
                 freq=10000,
                 pos_enc=None,
                 num_segments=None,
                 heads=1,
                 fusion=None):
        """ Class wrapping the MultiAttention part of PGL-SUM; its key modules and parameters.

        :param int input_size: The expected input feature size.
        :param int output_size: The hidden feature size of the attention mechanisms.
        :param int freq: The frequency of the sinusoidal positional encoding.
        :param None | str pos_enc: The selected positional encoding [absolute, relative].
        :param None | int num_segments: The selected number of segments to split the videos.
        :param int heads: The selected number of global heads.
        :param None | str fusion: The selected type of feature fusion.
        """
        super(MultiAttention, self).__init__()

        # Global Attention, considering differences among all frames
        self.attention = SelfAttention(
            input_size=input_size,
            output_size=output_size,
            freq=freq,
            pos_enc=pos_enc,
            heads=heads)

        self.num_segments = num_segments
        if self.num_segments is not None:
            assert self.num_segments >= 2, 'num_segments must be None or 2+'
            self.local_attention = nn.ModuleList()
            for _ in range(self.num_segments):
                # Local Attention, considering differences among the same segment with reduce hidden size
                self.local_attention.append(
                    SelfAttention(
                        input_size=input_size,
                        output_size=output_size // num_segments,
                        freq=freq,
                        pos_enc=pos_enc,
                        heads=4))
        self.permitted_fusions = ['add', 'mult', 'avg', 'max']
        self.fusion = fusion
        if self.fusion is not None:
            self.fusion = self.fusion.lower()
            assert self.fusion in self.permitted_fusions, f'Fusion method must be: {*self.permitted_fusions,}'

    def forward(self, x):
        """ Compute the weighted frame features, based on the global and locals (multi-head) attention mechanisms.

        :param torch.Tensor x: Tensor with shape [T, input_size] containing the frame features.
        :return: A tuple of:
            weighted_value: Tensor with shape [T, input_size] containing the weighted frame features.
            attn_weights: Tensor with shape [T, T] containing the attention weights.
        """
        weighted_value, attn_weights = self.attention(x)  # global attention

        if self.num_segments is not None and self.fusion is not None:
            segment_size = math.ceil(x.shape[0] / self.num_segments)
            for segment in range(self.num_segments):
                left_pos = segment * segment_size
                right_pos = (segment + 1) * segment_size
                local_x = x[left_pos:right_pos]
                weighted_local_value, attn_local_weights = self.local_attention[
                    segment](local_x)  # local attentions

                # Normalize the features vectors
                weighted_value[left_pos:right_pos] = F.normalize(
                    weighted_value[left_pos:right_pos].clone(), p=2, dim=1)
                weighted_local_value = F.normalize(
                    weighted_local_value, p=2, dim=1)
                if self.fusion == 'add':
                    weighted_value[left_pos:right_pos] += weighted_local_value
                elif self.fusion == 'mult':
                    weighted_value[left_pos:right_pos] *= weighted_local_value
                elif self.fusion == 'avg':
                    weighted_value[left_pos:right_pos] += weighted_local_value
                    weighted_value[left_pos:right_pos] /= 2
                elif self.fusion == 'max':
                    weighted_value[left_pos:right_pos] = torch.max(
                        weighted_value[left_pos:right_pos].clone(),
                        weighted_local_value)

        return weighted_value, attn_weights


class PGL_SUM(nn.Module):

    def __init__(self,
                 input_size=1024,
                 output_size=1024,
                 freq=10000,
                 pos_enc=None,
                 num_segments=None,
                 heads=1,
                 fusion=None):
        """ Class wrapping the PGL-SUM model; its key modules and parameters.

        :param int input_size: The expected input feature size.
        :param int output_size: The hidden feature size of the attention mechanisms.
        :param int freq: The frequency of the sinusoidal positional encoding.
        :param None | str pos_enc: The selected positional encoding [absolute, relative].
        :param None | int num_segments: The selected number of segments to split the videos.
        :param int heads: The selected number of global heads.
        :param None | str fusion: The selected type of feature fusion.
        """
        super(PGL_SUM, self).__init__()

        self.attention = MultiAttention(
            input_size=input_size,
            output_size=output_size,
            freq=freq,
            pos_enc=pos_enc,
            num_segments=num_segments,
            heads=heads,
            fusion=fusion)
        self.linear_1 = nn.Linear(
            in_features=input_size, out_features=input_size)
        self.linear_2 = nn.Linear(
            in_features=self.linear_1.out_features, out_features=1)

        self.drop = nn.Dropout(p=0.5)
        self.norm_y = nn.LayerNorm(normalized_shape=input_size, eps=1e-6)
        self.norm_linear = nn.LayerNorm(
            normalized_shape=self.linear_1.out_features, eps=1e-6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, frame_features):
        """ Produce frames importance scores from the frame features, using the PGL-SUM model.

        :param torch.Tensor frame_features: Tensor of shape [T, input_size] containing the frame features produced by
        using the pool5 layer of GoogleNet.
        :return: A tuple of:
            y: Tensor with shape [1, T] containing the frames importance scores in [0, 1].
            attn_weights: Tensor with shape [T, T] containing the attention weights.
        """
        frame_features = frame_features.reshape(-1, frame_features.shape[-1])
        residual = frame_features
        weighted_value, attn_weights = self.attention(frame_features)
        y = weighted_value + residual
        y = self.drop(y)
        y = self.norm_y(y)

        # 2-layer NN (Regressor Network)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.drop(y)
        y = self.norm_linear(y)

        y = self.linear_2(y)
        y = self.sigmoid(y)
        y = y.view(1, -1)

        return y, attn_weights
