import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
from typing import Optional, Tuple, List
from timm.models.layers import to_2tuple, _assert, Mlp, DropPath

class Patchify(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            in_chs: int = 3,
            out_chs: int = 64,
            kernel_size: int = 3,
            stride: int = 2,
            flatten=True,
            hidden=False,
            bias=False,
    ):
        super().__init__()
        self.flatten = flatten
        self.hidden = hidden
        padding = lambda kernel_size, stride: (kernel_size - stride + 1) // 2  # padding(kernel_size, stride)
        self.conv1 = nn.Conv2d(in_chs, out_chs // 2, kernel_size=kernel_size, stride=stride, padding=padding(kernel_size, stride), bias=bias)
        self.norm1 = nn.BatchNorm2d(out_chs // 2)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_chs // 2, out_chs, kernel_size=kernel_size, stride=1, padding=padding(kernel_size, 1), bias=bias)

    def forward(self, x):
        # B, T, C, H, W = x.shape   [128, 256, 28, 28]
        x = self.conv1(x)         # [128, 128, 14, 14]
        # print(x.shape)
        x = self.norm1(x)         # [128, 128, 14, 14]
        # print(x.shape)
        hidden = x = self.act1(x)
        x = self.conv2(x)         # [128, 256, 14, 14]
        # print(x.shape)

        _, C, H, W = x.shape    # [128, 256, 14, 14]

        if self.flatten:
            x = rearrange(x, 'bt c h w -> bt (h w) c')  # (b t) c h w -> (b t) n c  [128, 196, 128])

        if self.hidden:
            return x, (H, W), hidden
        else:
            return x, (H, W)

class UnPatchify(nn.Module):
    def __init__(
            self,
            in_chs: int = 64,
            out_chs: int = 3,
            kernel_size: int = 3,
            stride: int = 2,
            flatten=True,
            bias=False,
    ):
        super().__init__()
        padding = lambda kernel_size, stride: (kernel_size - stride + 1) // 2
        self.flatten = flatten
        self.conv1 = nn.Conv2d(in_chs, in_chs // 2, kernel_size=kernel_size, stride=1, padding=padding(kernel_size, 1), bias=bias)
        self.norm1 = nn.BatchNorm2d(in_chs // 2)
        self.act1 = nn.GELU()
        if stride == 2:
            self.conv2 = nn.Sequential(*[
                nn.Conv2d(in_chs // 2, out_chs * 4, kernel_size=kernel_size,
                          stride=1, padding=padding(kernel_size, 1), bias=bias),
                nn.PixelShuffle(2)
            ])
        else:
            self.conv2 = nn.Conv2d(in_chs // 2, out_chs, kernel_size=kernel_size,
                          stride=1, padding=padding(kernel_size, 1), bias=bias)

    def forward(self, x, t=None, size=None, hidden=None):
        # B, T, C, H, W = x.shape
        if self.flatten:
            x = rearrange(x, 'bt (h w) c -> bt c h w', h=size[0], w=size[1])
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        if hidden is not None:
            x = x + hidden
        x = self.conv2(x)
        return x

class SpatialGate(nn.Module):
    """ Spatial-Gate.
    Args:
        dim (int): Half of input channels.
    """
    def __init__(self, dim):
        super().__init__()
        x2_dim = dim // 2
        self.norm = nn.LayerNorm(x2_dim)
        self.conv = nn.Conv2d(x2_dim, x2_dim, kernel_size=3, stride=1, padding=1, groups=x2_dim)  # DW Conv

    def forward(self, x, size=None):
        # Split
        x1, x2 = x.chunk(2, dim=-1)
        x2 = self.norm(x2)
        if len(x.shape) == 3:
            x2 = rearrange(x2, 'bt (h w) c -> bt c h w', h=size[0], w=size[1])
            x2 = self.conv(x2)
            x2 = rearrange(x2, 'bt c h w -> bt (h w) c')
        else:
            x2 = rearrange(x2, 'bt h w c -> bt c h w')
            x2 = self.conv(x2)
            x2 = rearrange(x2, 'bt c h w -> bt h w c')
        return x1 * x2

class GateFFN(Mlp):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__(in_features, hidden_features=hidden_features, out_features=out_features, act_layer=act_layer, drop=drop)
        self.SpatialGate = SpatialGate(dim=hidden_features)
        self.fc2 = nn.Linear(hidden_features // 2, in_features)

    def forward(self, x, size=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.SpatialGate(x, size)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class CasualTimeAttention(nn.Module):
    """
        S1, S2 = N, p1 * p2
        casual_mask = torch.tril(torch.ones(S1, S1)).cuda()  # S1 is total matrix size, N is step length
        for i in range(0, S1, S2):
            casual_mask[i:i + S2, i:i + S2] = 1
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def split_multi_heads(self, x):
        return rearrange(x, 'b n (num_head head_dim) -> b num_head n head_dim', num_head=self.num_heads)

    def merge_multi_heads(self, x):
        return rearrange(x, 'b num_head n head_dim -> b n (num_head head_dim)')

    def forward(self, x, t=None, size=None):
        # BT, N, C = x.shape
        _, n, _ = x.shape
        x = rearrange(x, '(b t) n c -> (b n) t c', t=t)
        B, N, C = x.shape  # bn as B, t as N, c as C

        casual_mask = torch.tril(torch.ones(B, self.num_heads, N, N)).cuda()  # upper triangle set zero

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(casual_mask == 0, float('-inf')).softmax(dim=-1)  # upper triangle mask out '-inf'
        # attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)   # (b n) t c
        x = rearrange(x, '(b n) t c -> (b t) n c', n=n)
        return x

class CasualTimeBlock(nn.Module):
    r""" Channel-wise Local Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        ffn (bool): If False, pure attention network without FFNs
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CasualTimeAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.ls1 = LayerScale(dim, init_values=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer)

        self.ls2 = LayerScale(dim, init_values=1e-6)

    def forward(self, x, t=None, size=None):
        # bt, n, c = x.shape
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), t=t, size=size)))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x

def grid_partition(x, grid_size: List[int]):
    B, H, W, C = x.shape
    _assert(H % grid_size[0] == 0, f'height {H} must be divisible by grid {grid_size[0]}')
    _assert(W % grid_size[1] == 0, '')
    x = x.view(B, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1], C)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, grid_size[0], grid_size[1], C)
    return windows

def grid_reverse(windows, grid_size: List[int], img_size: List[int]):
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, H, W, C)
    return x


class RelPosMlp(nn.Module):
    """ Log-Coordinate Relative Position MLP
    Based on ideas presented in Swin-V2 paper (https://arxiv.org/abs/2111.09883)

    This impl covers the 'swin' implementation as well as two timm specific modes ('cr', and 'rw')
    """
    def __init__(
            self,
            window_size,
            num_heads=8,
            hidden_dim=128,
            prefix_tokens=0,
            mode='cr',
            pretrained_window_size=(0, 0)
    ):
        super().__init__()
        self.window_size = window_size
        self.window_area = self.window_size[0] * self.window_size[1]
        self.prefix_tokens = prefix_tokens
        self.num_heads = num_heads
        self.bias_shape = (self.window_area,) * 2 + (num_heads,)
        if mode == 'swin':
            self.bias_act = nn.Sigmoid()
            self.bias_gain = 16
            mlp_bias = (True, False)
        elif mode == 'rw':
            self.bias_act = nn.Tanh()
            self.bias_gain = 4
            mlp_bias = True
        else:
            self.bias_act = nn.Identity()
            self.bias_gain = None
            mlp_bias = True

        self.mlp = Mlp(
            2,  # x, y
            hidden_features=hidden_dim,
            out_features=num_heads,
            act_layer=nn.ReLU,
            drop=(0.125, 0.)
        )

        self.register_buffer(
            "relative_position_index",
            gen_relative_position_index(window_size),
            persistent=False)

        # get relative_coords_table
        self.register_buffer(
            "rel_coords_log",
            gen_relative_log_coords(window_size, pretrained_window_size, mode=mode),
            persistent=False)

    def get_bias(self) -> torch.Tensor:
        relative_position_bias = self.mlp(self.rel_coords_log)
        if self.relative_position_index is not None:
            relative_position_bias = relative_position_bias.view(-1, self.num_heads)[
                self.relative_position_index.view(-1)]  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.view(self.bias_shape)
        relative_position_bias = relative_position_bias.permute(2, 0, 1)
        relative_position_bias = self.bias_act(relative_position_bias)
        if self.bias_gain is not None:
            relative_position_bias = self.bias_gain * relative_position_bias
        if self.prefix_tokens:
            relative_position_bias = F.pad(relative_position_bias, [self.prefix_tokens, 0, self.prefix_tokens, 0])
        return relative_position_bias.unsqueeze(0).contiguous()

    def forward(self, attn, shared_rel_pos: Optional[torch.Tensor] = None):
        return attn + self.get_bias()


def gen_relative_position_index(
        q_size: Tuple[int, int],
        k_size: Tuple[int, int] = None,
        class_token: bool = False) -> torch.Tensor:
    # Adapted with significant modifications from Swin / BeiT codebases
    # get pair-wise relative position index for each token inside the window
    q_coords = torch.stack(torch.meshgrid([torch.arange(q_size[0]), torch.arange(q_size[1])])).flatten(1)  # 2, Wh, Ww
    if k_size is None:
        k_coords = q_coords
        k_size = q_size
    else:
        # different q vs k sizes is a WIP
        k_coords = torch.stack(torch.meshgrid([torch.arange(k_size[0]), torch.arange(k_size[1])])).flatten(1)
    relative_coords = q_coords[:, :, None] - k_coords[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0)  # Wh*Ww, Wh*Ww, 2
    _, relative_position_index = torch.unique(relative_coords.view(-1, 2), return_inverse=True, dim=0)

    if class_token:
        # handle cls to token & token 2 cls & cls to cls as per beit for rel pos bias
        # NOTE not intended or tested with MLP log-coords
        max_size = (max(q_size[0], k_size[0]), max(q_size[1], k_size[1]))
        num_relative_distance = (2 * max_size[0] - 1) * (2 * max_size[1] - 1) + 3
        relative_position_index = F.pad(relative_position_index, [1, 0, 1, 0])
        relative_position_index[0, 0:] = num_relative_distance - 3
        relative_position_index[0:, 0] = num_relative_distance - 2
        relative_position_index[0, 0] = num_relative_distance - 1

    return relative_position_index.contiguous()


def gen_relative_log_coords(
        win_size: Tuple[int, int],
        pretrained_win_size: Tuple[int, int] = (0, 0),
        mode='swin',
):
    assert mode in ('swin', 'cr', 'rw')
    # as per official swin-v2 impl, supporting timm specific 'cr' and 'rw' log coords as well
    relative_coords_h = torch.arange(-(win_size[0] - 1), win_size[0], dtype=torch.float32)
    relative_coords_w = torch.arange(-(win_size[1] - 1), win_size[1], dtype=torch.float32)
    relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))
    relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous()  # 2*Wh-1, 2*Ww-1, 2
    if mode == 'swin':
        if pretrained_win_size[0] > 0:
            relative_coords_table[:, :, 0] /= (pretrained_win_size[0] - 1)
            relative_coords_table[:, :, 1] /= (pretrained_win_size[1] - 1)
        else:
            relative_coords_table[:, :, 0] /= (win_size[0] - 1)
            relative_coords_table[:, :, 1] /= (win_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            1.0 + relative_coords_table.abs()) / math.log2(8)
    else:
        if mode == 'rw':
            # cr w/ window size normalization -> [-1,1] log coords
            relative_coords_table[:, :, 0] /= (win_size[0] - 1)
            relative_coords_table[:, :, 1] /= (win_size[1] - 1)
            relative_coords_table *= 8  # scale to -8, 8
            relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                1.0 + relative_coords_table.abs())
            relative_coords_table /= math.log2(9)   # -> [-1, 1]
        else:
            # mode == 'cr'
            relative_coords_table = torch.sign(relative_coords_table) * torch.log(
                1.0 + relative_coords_table.abs())

    return relative_coords_table

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma
        return x.mul_(gamma) if self.inplace else x * gamma

class GridUnshuffleAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., grid_size=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = self.dim_head ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.partition_size = to_2tuple(grid_size)
        self.rel_pos = RelPosMlp(window_size=grid_size, num_heads=num_heads, hidden_dim=512)

    def attn(self, x):
        B = x.shape[0]
        restore_shape = x.shape[:-1]

        q, k, v = self.qkv(x).view(B, -1, self.num_heads, self.dim_head * 3).transpose(1, 2).chunk(3, dim=3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.rel_pos is not None:
            attn = self.rel_pos(attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(restore_shape + (-1,))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x, size=None):  # t=total frame, size=(H, W)
        partitioned = grid_partition(x, self.partition_size) # (b t) h w c
        partitioned = self.attn(partitioned)
        x = grid_reverse(partitioned, self.partition_size, size)
        return x   # (b t) h w c

class GridUnshuffleBlock(nn.Module):
    r""" Channel-wise Local Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        ffn (bool): If False, pure attention network without FFNs
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, grid_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GridUnshuffleAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            grid_size=to_2tuple(grid_size)
        )
        self.ls1 = LayerScale(dim, init_values=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer)

        self.ls2 = LayerScale(dim, init_values=1e-6)

    def forward(self, x, size=None):
        # BN, T, C = x.shape
        # [128, 196, 128])
        x = rearrange(x, 'bt (h w) c -> bt h w c', h=size[0], w=size[1])
        # [128, 14, 14, 128]
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), size=size)))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        x = rearrange(x, 'bt h w c -> bt (h w) c')  # bt n c
        return x

class ChannelAttention(nn.Module):
    r""" Channel based self attention.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of the groups.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = k * self.scale
        attention = k.transpose(-1, -2) @ v
        attention = attention.softmax(dim=-1)
        x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class ConvPosEnc(nn.Module):
    """Depth-wise convolution to get the positional information.
    """
    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim,
                              dim,
                              to_2tuple(k),
                              to_2tuple(1),
                              to_2tuple(k // 2),
                              groups=dim)

    def forward(self, x, size: Tuple[int, int]):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.proj(feat)
        feat = feat.flatten(2).transpose(1, 2)
        x = x + feat
        return x


class GroupChannelBlock(nn.Module):
    r""" Channel-wise Local Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        ffn (bool): If False, pure attention network without FFNs
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True):
        super().__init__()

        self.cpe = nn.ModuleList([ConvPosEnc(dim=dim, k=3),
                                  ConvPosEnc(dim=dim, k=3)])
        self.ffn = ffn
        self.norm1 = norm_layer(dim)
        self.attn = ChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)

    def forward(self, x, size=None):
        x = self.cpe[0](x, size)
        cur = self.norm1(x)
        cur = self.attn(cur)
        x = x + self.drop_path(cur)

        x = self.cpe[1](x, size)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TripletBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 grid_size=8):
        super().__init__()

        self.CasualTimeBlock = CasualTimeBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_path=drop_path,
                                               act_layer=act_layer, norm_layer=norm_layer)
        self.GridUnshuffleBlock = GridUnshuffleBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_path=drop_path,
                                                           act_layer=act_layer, norm_layer=norm_layer, grid_size=grid_size)
        self.GroupChannelBlock = GroupChannelBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_path=drop_path,
                                         act_layer=act_layer, norm_layer=norm_layer, ffn=True)


    def forward(self, x, t=None, size=None):  # bt n c
        x = self.CasualTimeBlock(x, t=t, size=size)   # bt n c
        x = self.GridUnshuffleBlock(x, size=size)  # bt n c
        x = self.GroupChannelBlock(x, size=size)  # bt n c
        return x

class TAMSTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 grid_size=8):
        super().__init__()

        self.CasualTimeBlock = CasualTimeBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_path=drop_path,
                                               act_layer=act_layer, norm_layer=norm_layer)
        self.GridUnshuffleBlock = GridUnshuffleBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_path=drop_path,
                                                           act_layer=act_layer, norm_layer=norm_layer, grid_size=grid_size)


    def forward(self, x, t=None, size=None):  # bt n c
        x = self.CasualTimeBlock(x, t=t, size=size)   # bt n c
        x = self.GridUnshuffleBlock(x, size=size)  # bt n c
        return x

class TAMSCBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 grid_size=8):
        super().__init__()


        self.GridUnshuffleBlock = GridUnshuffleBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_path=drop_path,
                                                           act_layer=act_layer, norm_layer=norm_layer, grid_size=grid_size)
        self.GroupChannelBlock = GroupChannelBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_path=drop_path,
                                         act_layer=act_layer, norm_layer=norm_layer, ffn=True)


    def forward(self, x, t=None, size=None):  # bt n c
        x = self.GroupChannelBlock(x, size=size)  # bt n c
        x = self.GridUnshuffleBlock(x, size=size)  # bt n c

        return x


class Triplet_Model_MMNIST(nn.Module):
    def __init__(self, in_shape, out_chs=128, num_blocks=3, num_heads=8, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, grid_size=8, max_frame=50, **kwargs):
        super().__init__()
        T, C, H, W = in_shape
        in_chs = C
        self.num_blocks = num_blocks
        self.patchify2 = Patchify(in_chs=in_chs, out_chs=out_chs, kernel_size=3, flatten=False, bias=False)
        self.patchify1 = Patchify(in_chs=out_chs, out_chs=out_chs, kernel_size=3, flatten=True, bias=False)

        stride = 4  # mmnist
        self.num_tokens = (H // stride) * (W // stride)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, out_chs))
        self.temporal_embed = nn.Parameter(torch.zeros(1, max_frame, out_chs))

        self.blocks = nn.ModuleList([])
        for i in range(num_blocks):
            block = TripletBlock(dim=out_chs, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                 drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, grid_size=grid_size)
            self.blocks.append(block)

        self.unpatchify1 = UnPatchify(in_chs=out_chs, out_chs=out_chs, kernel_size=3, flatten=True, bias=False)
        self.unpatchify2 = UnPatchify(in_chs=out_chs, out_chs=in_chs, kernel_size=3, flatten=False, bias=False)
 
    def update_drop_path(self, drop_path_rate):
        dp_list = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_blocks * 3)]
        dp_lists = [dp_list[i:i + 3] for i in range(0, len(dp_list), 3)]
        for block, dp_rate in zip(self.blocks, dp_lists):
            block.CasualTimeBlock.drop_path.drop_prob = dp_rate[0]
            block.GridUnshuffleBlock.drop_path.drop_prob = dp_rate[1]
            block.GroupChannelBlock.drop_path.drop_prob = dp_rate[2]


    def forward(self, x):
        b, t, c, h, w = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x, size2 = self.patchify2(x)  # bt n c
        x, size1 = self.patchify1(x)  # bt n c

        x = x + self.pos_embed

        x = rearrange(x, '(b t) n c -> (b n) t c', b=b)
        x = x + self.temporal_embed[:, :t, :]
        x = rearrange(x, '(b n) t c -> (b t) n c', b=b)

        for block in self.blocks:
            x = block(x, t=t, size=size1)

        x = self.unpatchify1(x, t=t, size=size1)
        x = self.unpatchify2(x, t=t, size=size2)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=t)
        return x


class Triplet_Model_Taxibj(nn.Module):
    def __init__(self, in_shape, out_chs=128, num_blocks=3, num_heads=8, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, stride=2, grid_size=8, max_frame=50, **kwargs):
        super().__init__()
        T, C, H, W = in_shape
        in_chs = C
        self.num_blocks = num_blocks

        self.patchify = Patchify(in_chs=in_chs, out_chs=out_chs, kernel_size=3, stride=stride, flatten=True, bias=False)

        self.num_tokens = (H // stride) * (W // stride)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, out_chs))
        self.temporal_embed = nn.Parameter(torch.zeros(1, max_frame, out_chs))

        self.blocks = nn.ModuleList([])
        for i in range(num_blocks):
            block = TripletBlock(dim=out_chs, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                 drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, grid_size=grid_size)
            self.blocks.append(block)

        self.unpatchify = UnPatchify(in_chs=out_chs, out_chs=in_chs, kernel_size=3, stride=stride, flatten=True, bias=False)

    def update_drop_path(self, drop_path_rate):
        dp_list = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_blocks * 3)]
        dp_lists = [dp_list[i:i + 3] for i in range(0, len(dp_list), 3)]
        for block, dp_rate in zip(self.blocks, dp_lists):
            block.CasualTimeBlock.drop_path.drop_prob = dp_rate[0]
            block.GridUnshuffleBlock.drop_path.drop_prob = dp_rate[1]
            block.GroupChannelBlock.drop_path.drop_prob = dp_rate[2]


    def forward(self, x):
        b, t, c, h, w = x.shape     # [16, 256, 8, 28, 28]
        x = rearrange(x, 'b t c h w -> (b t) c h w')    # [128, 256, 28, 28]
        x, size = self.patchify(x)  # bt n c = [128, 196, 128]

        x = x + self.pos_embed

        x = rearrange(x, '(b t) n c -> (b n) t c', b=b)    
        x = x + self.temporal_embed[:, :t, :]
        x = rearrange(x, '(b n) t c -> (b t) n c', b=b)     # [(16 8) 196 128], b=16

        for block in self.blocks:
            x = block(x, t=t, size=size)

        x = self.unpatchify(x, t=t, size=size)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=t)
        return x



