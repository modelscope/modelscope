# Part of the implementation is borrowed and modified from 2s-AGCN, publicly available at
# https://github.com/lshiwjx/2s-AGCN
import math

import torch
import torch.nn as nn
from einops import rearrange


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def zero(x):
    """return zero."""
    return 0


def iden(x):
    """return input itself."""
    return x


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.,
                 changedim=False,
                 currentdim=0,
                 depth=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 comb=False,
                 vis=False):
        """Attention is all you need

        Args:
            dim (_type_): _description_
            num_heads (int, optional): _description_. Defaults to 8.
            qkv_bias (bool, optional): _description_. Defaults to False.
            qk_scale (_type_, optional): _description_. Defaults to None.
            attn_drop (_type_, optional): _description_. Defaults to 0..
            proj_drop (_type_, optional): _description_. Defaults to 0..
            comb (bool, optional): Defaults to False.
                True: q transpose * k.
                False: q * k transpose.
            vis (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb
        self.vis = vis

    def forward(self, fv, fe):
        B, N, C = fv.shape
        B, E, C = fe.shape
        q = self.to_q(fv).reshape(B, N, self.num_heads,
                                  C // self.num_heads).permute(0, 2, 1, 3)
        k = self.to_k(fe).reshape(B, E, self.num_heads,
                                  C // self.num_heads).permute(0, 2, 1, 3)
        v = self.to_v(fe).reshape(B, E, self.num_heads,
                                  C // self.num_heads).permute(0, 2, 1, 3)
        # Now fv shape (B, H, N, C//heads)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if self.comb:
            fv = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
            fv = rearrange(fv, 'B H N C -> B N (H C)')
        elif self.comb is False:
            fv = (attn @ v).transpose(1, 2).reshape(B, N, C)
        fv = self.proj(fv)
        fv = self.proj_drop(fv)
        return fv


class FirstOrderAttention(nn.Module):
    """First Order Attention block for spatial relationship.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 A,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 adj_len=17,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.A = A
        self.PA = nn.Parameter(torch.FloatTensor(3, adj_len, adj_len))
        torch.nn.init.constant_(self.PA, 1e-6)

        self.num_subset = 3
        inter_channels = out_channels // 4
        self.inter_c = inter_channels
        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        self.linears = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))
            self.linears.append(nn.Linear(in_channels, in_channels))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels))
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        assert self.A.shape[0] == self.kernel_size[1]

        N, C, T, V = x.size()
        A = self.A + self.PA

        y = None
        for i in range(self.num_subset):
            x_in = rearrange(x, 'N C T V -> N T V C')
            x_in = self.linears[i](x_in)
            A0 = rearrange(x_in, 'N T V C -> N (C T) V')

            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(
                N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))
            A1 = A1 + A[i]
            z = self.conv_d[i](torch.matmul(A0, A1).view(N, C, T, V))
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)

        return self.relu(y)


class HightOrderAttentionBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 A,
                 di_graph,
                 attention=False,
                 stride=1,
                 adj_len=17,
                 dropout=0,
                 residual=True,
                 norm_layer=nn.BatchNorm2d,
                 edge_importance=False,
                 graph=None,
                 conditional=False,
                 experts=4,
                 bias=True,
                 share_tcn=False,
                 max_hop=2):
        super().__init__()

        t_kernel_size = kernel_size[0]
        assert t_kernel_size % 2 == 1
        padding = ((t_kernel_size - 1) // 2, 0)
        self.max_hop = max_hop
        self.attention = attention
        self.di_graph = di_graph

        self.foa_block = FirstOrderAttention(
            in_channels,
            out_channels,
            kernel_size,
            A,
            bias=bias,
            adj_len=adj_len)

        self.tcn_v = nn.Sequential(
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels, (t_kernel_size, 1), (stride, 1),
                padding,
                bias=bias),
            norm_layer(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual_v = zero
        elif (in_channels == out_channels) and (stride == 1):
            self.residual_v = iden
        else:
            self.residual_v = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=bias),
                norm_layer(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

        if self.attention:
            self.cross_attn = Attention(
                dim=out_channels,
                num_heads=8,
                qkv_bias=True,
                qk_scale=None,
                attn_drop=dropout,
                proj_drop=dropout)
            self.norm_v = nn.LayerNorm(out_channels)
            self.mlp = Mlp(
                in_features=out_channels,
                out_features=out_channels,
                hidden_features=out_channels * 2,
                act_layer=nn.GELU,
                drop=dropout)
            self.norm_mlp = nn.LayerNorm(out_channels)

            # linear to change fep channels
            self.linears = nn.ModuleList()
            for hop_i in range(self.max_hop - 1):
                hop_linear = nn.ModuleList()
                for i in range(
                        len(
                            eval(f'self.di_graph.directed_edges_hop{hop_i+2}'))
                ):
                    hop_linear.append(nn.Linear(hop_i + 2, 1))
                self.linears.append(hop_linear)

    def forward(self, fv, fe):
        # `fv` (node features) has shape (B, C, T, V_node)
        # `fe` (edge features) has shape (B, C, T, V_edge)
        N, C, T, V = fv.size()

        res_v = self.residual_v(fv)

        fvp = self.foa_block(fv)
        fep_out = (
            fvp[..., [c for p, c in self.di_graph.directed_edges_hop1]]
            - fvp[..., [p for p, c in self.di_graph.directed_edges_hop1]]
        ).contiguous()

        if self.attention:
            fep_concat = None
            for hop_i in range(self.max_hop):
                if 0 == hop_i:
                    fep_hop_i = (fvp[..., [
                        c for p, c in eval(
                            f'self.di_graph.directed_edges_hop{hop_i+1}')
                    ]] - fvp[..., [
                        p for p, c in eval(
                            f'self.di_graph.directed_edges_hop{hop_i+1}')
                    ]]).contiguous()
                    fep_hop_i = rearrange(fep_hop_i, 'N C T E -> (N T) E C')
                else:
                    joints_parts = eval(
                        f'self.di_graph.directed_edges_hop{hop_i+1}')
                    fep_hop_i = None
                    for part_idx, part in enumerate(joints_parts):
                        fep_part = None
                        for j in range(len(part) - 1):
                            fep = (fvp[..., part[j + 1]]
                                   - fvp[..., part[j]]).contiguous().unsqueeze(
                                       dim=-1)
                            if fep_part is None:
                                fep_part = fep
                            else:
                                fep_part = torch.cat((fep_part, fep), dim=-1)
                        fep_part = self.linears[hop_i - 1][part_idx](fep_part)
                        if fep_hop_i is None:
                            fep_hop_i = fep_part
                        else:
                            fep_hop_i = torch.cat((fep_hop_i, fep_part),
                                                  dim=-1)

                    fep_hop_i = rearrange(fep_hop_i, 'N C T E -> (N T) E C')

                if fep_concat is None:
                    fep_concat = fep_hop_i
                else:
                    fep_concat = torch.cat((fep_concat, fep_hop_i),
                                           dim=-2)  # dim=-2 represent edge dim
            fvp = rearrange(fvp, 'N C T V -> (N T) V C')
            fvp = self.norm_v(self.cross_attn(fvp, fep_concat)) + iden(fvp)
            fvp = self.mlp(self.norm_mlp(fvp)) + iden(
                fvp)  # make output joint number = adj_len
            fvp = rearrange(fvp, '(N T) V C -> N C T V', N=N)

        fvp = self.tcn_v(fvp) + res_v

        return self.relu(fvp), fep_out
