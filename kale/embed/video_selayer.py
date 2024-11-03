# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@outlook.com
# =============================================================================

"""Python implementation of Squeeze-and-Excitation Layers (SELayer)
Initial implementation: channel-wise (SELayerC)
Improved implementation: temporal-wise (SELayerT), convolution-based channel-wise (SELayerCoC), max-pooling-based
channel-wise (SELayerMC), multi-pooling-based channel-wise (SELayerMAC)

[Redundancy and repeat of code will be reduced in the future.]

References:
    Hu Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." In CVPR, pp. 7132-7141. 2018.
    For initial implementation, please go to https://github.com/hujie-frank/SENet
"""

import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
from typing import Optional, Tuple, List
from timm.models.layers import to_2tuple, _assert, Mlp, DropPath
from kale.embed.video_TAM import *


def get_selayer(attention):
    """Get SELayers referring to attention.

    Args:
        attention (string): the name of the SELayer.
            (Options: ["SELayerC", "SELayerT", "SRMLayerVideo", "CSAMLayer", "STAMLayer",
            "SELayerCoC", "SELayerMC", "SELayerMAC"])

    Returns:
        se_layer (SELayer, optional): the SELayer.
    """

    if attention == "SELayerC":
        se_layer = SELayerC
    elif attention == "SELayerT":
        se_layer = SELayerT
    elif attention == "SRMLayerVideo":
        se_layer = SRMLayerVideo
    elif attention == "CSAMLayer":
        se_layer = CSAMLayer
    elif attention == "STAMLayer":
        se_layer = STAMLayer
    elif attention == "SELayerCoC":
        se_layer = SELayerCoC
    elif attention == "SELayerMC":
        se_layer = SELayerMC
    elif attention == "SELayerMAC":
        se_layer = SELayerMAC
    elif attention == "SELayerCoC":
        se_layer = SELayerCoC
    elif attention == "LayerCT":
        se_layer = LayerCT
    elif attention == "LayerST":
        se_layer = LayerST
    elif attention == "LayerTAM":
        se_layer = LayerTAM

    else:
        raise ValueError("Wrong MODEL.ATTENTION. Current:{}".format(attention))
    return se_layer


class SELayer(nn.Module):
    """Helper class for SELayer design."""

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.channel = channel
        self.reduction = reduction

    def forward(self, x):
        return NotImplementedError()


class SRMLayer(SELayer):
    """Construct Style-based Recalibration Module for images.

    References:
        Lee, HyunJae, Hyo-Eun Kim, and Hyeonseob Nam. "Srm: A style-based recalibration module for convolutional neural
        networks." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1854-1862. 2019.
    """

    def __init__(self, channel, reduction=16):
        # Reduction for compatibility with layer_block interface
        super(SRMLayer, self).__init__(channel, reduction)

        # CFC: channel-wise fully connected layer
        self.cfc = nn.Conv1d(self.channel, self.channel, kernel_size=2, bias=False, groups=self.channel)
        self.bn = nn.BatchNorm1d(self.channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        # Style pooling
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # Style integration
        z = self.cfc(u)  # (b, c, 1)
        z = self.bn(z)
        g = self.sigmoid(z)
        g = g.view(b, c, 1, 1)
        # out = x * g.expand_as(x)
        out = x + x * g.expand_as(x)
        return out


class SRMLayerVideo(SELayer):
    def __init__(self, channel, reduction=16):
        super(SRMLayerVideo, self).__init__(channel, reduction)
        self.cfc = nn.Conv1d(self.channel, self.channel, kernel_size=2, bias=False, groups=self.channel)
        self.bn = nn.BatchNorm1d(self.channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()

        # Style pooling
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # Style integration
        z = self.cfc(u)  # (b, c, 1)
        z = self.bn(z)
        g = self.sigmoid(z)
        g = g.view(b, c, 1, 1, 1)
        out = x + x * g.expand_as(x)
        return out



class LayerCT(SELayer):
    """Construct channel-temporal-wise SELayer."""

    def __init__(self, channel, reduction=16):
        super(LayerCT, self).__init__(channel, reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  
        self.conv1 = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, t, _, _ = x.size()                                # [16, 256, 8, 28, 28]
        y = self.avg_pool(x).view(b, c, t).unsqueeze(1)         # [16, 256, 8, 1, 1] -> [16, 256, 8] -> [16, 1, 256, 8]
        # print(f"y shape of avg_pool : {y.shape}")
        y = self.conv1(y)                                       # [16, 1, 256, 8]
        # print(f"y shape of conv1 : {y.shape}")
        y = self.relu(y)
        y = self.conv2(y)                                       # [16, 1, 256, 8]
        # print(f"y shape of conv2 : {y.shape}")
        y = self.sigmoid(y).squeeze(1).view(b, c, t, 1, 1)      # [16, 1, 256, 8] -> [16, 256, 8] -> [16, 256, 8, 1, 1]
        # print(f"y shape of sigmoid : {y.shape}")
        y = y - 0.5
        out = x + x * y.expand_as(x)                            # [16, 256, 8, 28, 28]
        # print(f"Output shape of LayerCT: {out.shape}")
        return out



class SELayerC(SELayer):
    """Construct channel-wise SELayer."""

    def __init__(self, channel, reduction=16):
        super(SELayerC, self).__init__(channel, reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _, _, _ = x.size()                # [16, 256, 8, 28, 28]
        y = self.avg_pool(x).view(b, c)         # [16, 256, 1, 1, 1] -> [16, 256]
        y = self.fc(y).view(b, c, 1, 1, 1)      # [16, 256] -> [16, 16] -> [16, 256] -> [16, 256, 1, 1, 1]
        # out = x * y.expand_as(x)
        y = y - 0.5
        out = x + x * y.expand_as(x)            # [16, 256, 8, 28, 28]
        # print(f"Output shape of SELayerC: {out.shape}")
        return out
        
class SELayerT(SELayer):
    """Construct temporal-wise SELayer."""

    def __init__(self, channel, reduction=2):
        super(SELayerT, self).__init__(channel, reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, _, t, _, _ = x.size()                    # [16, 256, 8, 28, 28]
        output = x.transpose(1, 2).contiguous()     # [16, 8, 256, 28, 28]
        y = self.avg_pool(output).view(b, t)        # [16, 8]
        y = self.fc(y).view(b, t, 1, 1, 1)          # [16, 8] -> [16, 4] -> [16, 8] -> [16, 8, 1, 1, 1]
        y = y.transpose(1, 2).contiguous()          # [16, 1, 8, 1, 1]
        # out = x * y.expand_as(x)
        y = y - 0.5                                 # [16, 1, 8, 1, 1]
        out = x + x * y.expand_as(x)                # [16, 256, 8, 28, 28]
        # print(f"Output shape of SELayerT: {out.shape}")
        return out


class LayerST(nn.Module):
    """Construct spatial-temporal-wise Layer."""

    def __init__(self, channel, reduction=2):
        super(LayerST, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool3d((1, None, None)) 
        self.conv1 = nn.Conv2d(self.channel, self.channel // self.reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.channel // self.reduction, self.channel, kernel_size=1, bias=False)
        # print(self.channel)
        # self.conv1 = nn.Conv2d(t,  t // self.reduction, kernel_size=1, bias=False) 
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(t // self.reduction, t, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, _, t, h, w = x.size()                                            # [16, 256, 8, 28, 28]
        x_reshaped = x.transpose(1, 2).contiguous()                         # [16, 8, 256, 28, 28]
        # print(f"LayerST y shape of reshaped x : {x_reshaped.shape}")
        y = self.avg_pool(x_reshaped).squeeze(2)                            # [16, 8, 1, 28, 28] -> [16, 8, 28, 28]
        # print(f"LayerST y shape of avg_pool : {y.shape}")
        y = self.conv1(y)                                                   # [16, 4, 28, 28]
        # print(f"LayerST y shape of conv1 : {y.shape}")
        y = self.relu(y)                                                    # [16, 4, 28, 28]
        y = self.conv2(y)                                                   # [16, 8, 28, 28]
        # print(f"LayerST y shape of conv2 : {y.shape}")
        y = self.sigmoid(y).unsqueeze(1).contiguous()                       # [16, 8, 28, 28] -> [16, 8, 1, 28, 28] -> [16, 1, 8, 28, 28]
        y = y - 0.5
        # print(f"LayerST y shape of unsqueezed : {y.shape}")
        out = x + x * y.expand_as(x)                                        # [16, 256, 8, 28, 28]
        # print(f"Output shape of SELayerT: {out.shape}")
        return out


class LayerTAM_ST(nn.Module):
    def __init__(self, C, H, out_chs=128, num_blocks=1, num_heads=8, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, stride=2, grid_size=1, max_frame=50, **kwargs):
        super().__init__()
        # T, C, H, W = in_shape
        # C, T, H, W = in_shape
        in_chs = C
        W = H
        out_chs = in_chs
        self.num_blocks = num_blocks

        self.patchify = Patchify(in_chs=in_chs, out_chs=out_chs, kernel_size=3, stride=stride, flatten=True, bias=False)

        self.num_tokens = (H // stride) * (W // stride)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, out_chs))
        self.temporal_embed = nn.Parameter(torch.zeros(1, max_frame, out_chs))

        self.blocks = nn.ModuleList([])
        for i in range(num_blocks):
            block = TAMSTBlock(dim=out_chs, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                 drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, grid_size=grid_size)
            self.blocks.append(block)

        self.unpatchify = UnPatchify(in_chs=out_chs, out_chs=in_chs, kernel_size=3, stride=stride, flatten=True, bias=False)

    def update_drop_path(self, drop_path_rate):
        dp_list = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_blocks * 3)]
        dp_lists = [dp_list[i:i + 3] for i in range(0, len(dp_list), 3)]
        for block, dp_rate in zip(self.blocks, dp_lists):
            block.CasualTimeBlock.drop_path.drop_prob = dp_rate[0]
            block.GridUnshuffleBlock.drop_path.drop_prob = dp_rate[1]



    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        b, t, c, h, w = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x, size = self.patchify(x)  # bt n c

        x = x + self.pos_embed

        x = rearrange(x, '(b t) n c -> (b n) t c', b=b)
        x = x + self.temporal_embed[:, :t, :]
        x = rearrange(x, '(b n) t c -> (b t) n c', b=b)

        for block in self.blocks:
            x = block(x, t=t, size=size)

        x = self.unpatchify(x, t=t, size=size)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=t)
        x = x.transpose(1, 2).contiguous()
        return x


class LayerTAM_CS(nn.Module):
    def __init__(self, C, H, out_chs=128, num_blocks=1, num_heads=8, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, stride=2, grid_size=1, max_frame=50, **kwargs):
        super().__init__()
        # T, C, H, W = in_shape
        # C, T, H, W = in_shape
        in_chs = C
        W = H
        out_chs = in_chs
        self.num_blocks = num_blocks

        self.patchify = Patchify(in_chs=in_chs, out_chs=out_chs, kernel_size=3, stride=stride, flatten=True, bias=False)

        self.num_tokens = (H // stride) * (W // stride)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, out_chs))
        self.temporal_embed = nn.Parameter(torch.zeros(1, max_frame, out_chs))

        self.blocks = nn.ModuleList([])
        for i in range(num_blocks):
            block = TAMSCBlock(dim=out_chs, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                 drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, grid_size=grid_size)
            self.blocks.append(block)

        self.unpatchify = UnPatchify(in_chs=out_chs, out_chs=in_chs, kernel_size=3, stride=stride, flatten=True, bias=False)

    def update_drop_path(self, drop_path_rate):
        dp_list = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_blocks * 3)]
        dp_lists = [dp_list[i:i + 3] for i in range(0, len(dp_list), 3)]
        for block, dp_rate in zip(self.blocks, dp_lists):
            block.GroupChannelBlock.drop_path.drop_prob = dp_rate[2]
            block.GridUnshuffleBlock.drop_path.drop_prob = dp_rate[1]



    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        b, t, c, h, w = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x, size = self.patchify(x)  # bt n c

        x = x + self.pos_embed

        x = rearrange(x, '(b t) n c -> (b n) t c', b=b)
        x = x + self.temporal_embed[:, :t, :]
        x = rearrange(x, '(b n) t c -> (b t) n c', b=b)

        for block in self.blocks:
            x = block(x, t=t, size=size)

        x = self.unpatchify(x, t=t, size=size)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=t)
        x = x.transpose(1, 2).contiguous()
        return x



class LayerTAM(nn.Module):
    def __init__(self, C, H, out_chs=128, num_blocks=1, num_heads=8, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, stride=2, grid_size=1, max_frame=50, **kwargs):
        super().__init__()
        # T, C, H, W = in_shape
        # C, T, H, W = in_shape
        in_chs = C
        W = H
        out_chs = in_chs
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
        x = x.transpose(1, 2).contiguous()
        b, t, c, h, w = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x, size = self.patchify(x)  # bt n c

        x = x + self.pos_embed

        x = rearrange(x, '(b t) n c -> (b n) t c', b=b)
        x = x + self.temporal_embed[:, :t, :]
        x = rearrange(x, '(b n) t c -> (b t) n c', b=b)

        for block in self.blocks:
            x = block(x, t=t, size=size)

        x = self.unpatchify(x, t=t, size=size)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=t)
        x = x.transpose(1, 2).contiguous()
        return x


# class LayerTAM(nn.Module):
#     def __init__(self, C, H, out_chs=128, num_blocks=1, num_heads=8, mlp_ratio=4., qkv_bias=False,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, grid_size=7, max_frame=50, **kwargs):
#         super().__init__()
#         # T, C, H, W = in_shape
#         in_chs = C
#         W = H
#         out_chs = in_chs
#         self.num_blocks = num_blocks
#         self.patchify2 = Patchify(in_chs=in_chs, out_chs=out_chs, kernel_size=3, flatten=False, bias=False)
#         self.patchify1 = Patchify(in_chs=out_chs, out_chs=out_chs, kernel_size=3, flatten=True, bias=False)

#         stride = 4  # mmnist
#         self.num_tokens = (H // stride) * (W // stride)
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, out_chs))
#         self.temporal_embed = nn.Parameter(torch.zeros(1, max_frame, out_chs))

#         self.blocks = nn.ModuleList([])
#         for i in range(num_blocks):
#             block = TripletBlock(dim=out_chs, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
#                  drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, grid_size=grid_size)
#             self.blocks.append(block)

#         self.unpatchify1 = UnPatchify(in_chs=out_chs, out_chs=out_chs, kernel_size=3, flatten=True, bias=False)
#         self.unpatchify2 = UnPatchify(in_chs=out_chs, out_chs=in_chs, kernel_size=3, flatten=False, bias=False)
 
#     def update_drop_path(self, drop_path_rate):
#         dp_list = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_blocks * 3)]
#         dp_lists = [dp_list[i:i + 3] for i in range(0, len(dp_list), 3)]
#         for block, dp_rate in zip(self.blocks, dp_lists):
#             block.CasualTimeBlock.drop_path.drop_prob = dp_rate[0]
#             block.GridUnshuffleBlock.drop_path.drop_prob = dp_rate[1]
#             block.GroupChannelBlock.drop_path.drop_prob = dp_rate[2]


#     def forward(self, x):
#         x = x.transpose(1, 2).contiguous()
#         b, t, c, h, w = x.shape
#         x = rearrange(x, 'b t c h w -> (b t) c h w')
#         x, size2 = self.patchify2(x)  # bt n c
#         x, size1 = self.patchify1(x)  # bt n c

#         x = x + self.pos_embed

#         x = rearrange(x, '(b t) n c -> (b n) t c', b=b)
#         x = x + self.temporal_embed[:, :t, :]
#         x = rearrange(x, '(b n) t c -> (b t) n c', b=b)

#         for block in self.blocks:
#             x = block(x, t=t, size=size1)

#         x = self.unpatchify1(x, t=t, size=size1)
#         x = self.unpatchify2(x, t=t, size=size2)
#         x = rearrange(x, '(b t) c h w -> b t c h w', t=t)
#         x = x.transpose(1, 2).contiguous()
#         return x


class CSAMLayer(nn.Module):
    """Construct Channel-Spatial Attention Module. This module [2] extends CBAM [1] by apply 3D layers.

    References:
        [1] Woo, Sanghyun, Jongchan Park, Joon-Young Lee, and In So Kweon. "Cbam: Convolutional block attention
        module." In Proceedings of the European conference on computer vision (ECCV), pp. 3-19. 2018.
        [2] Yi, Ziwen, Zhonghua Sun, Jinchao Feng, and Kebin Jia. "3D Residual Networks with Channel-Spatial Attention
        Module for Action Recognition." In 2020 Chinese Automation Congress (CAC), pp. 5171-5174. IEEE, 2020.
    """

    def __init__(self, channel, reduction=16):
        super(CSAMLayer, self).__init__()
        self.CAM = CSAMChannelModule(channel, reduction)
        self.SAM = CSAMSpatialModule()

    def forward(self, x):
        y = self.CAM(x)
        y = self.SAM(y)
        return y


class CSAMChannelModule(SELayer):
    def __init__(self, channel, reduction=16):
        super(CSAMChannelModule, self).__init__(channel, reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        y_avg = self.fc(y_avg).view(b, c, 1, 1, 1)
        y_max = self.fc(y_max).view(b, c, 1, 1, 1)
        y = torch.add(y_avg, y_max)
        y = self.sigmoid(y)
        out = x + x * y.expand_as(x)
        return out


class CSAMSpatialModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(CSAMSpatialModule, self).__init__()
        self.kernel_size = kernel_size
        self.compress = CSAMChannelPool()
        self.conv = nn.Conv3d(2, 1, self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)
        y = self.conv(x_compress)
        y = self.sigmoid(y)
        out = x + x * y.expand_as(x)
        return out


class CSAMChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class STAMLayer(SELayer):
    """Construct Spatial-temporal Attention Module.

    References:
        Zhou, Shengwei, Liang Bai, Haoran Wang, Zhihong Deng, Xiaoming Zhu, and Cheng Gong. "A Spatial-temporal
        Attention Module for 3D Convolution Network in Action Recognition." DEStech Transactions on Computer
        Science and Engineering cisnrc (2019).
    """

    def __init__(self, channel, reduction=16):
        super(STAMLayer, self).__init__(channel, reduction)
        self.kernel_size = 7
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel, bias=False),
        )
        self.conv = nn.Conv3d(1, 1, self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        y = self.sigmoid(y)
        y = x * y.expand_as(x)
        y = y.mean(1).unsqueeze(1)
        y = self.conv(y)
        y = self.sigmoid(y)
        out = x + x * y.expand_as(x)
        return out


class SELayerCoC(SELayer):
    """Construct convolution-based channel-wise SELayer."""

    def __init__(self, channel, reduction=16):
        super(SELayerCoC, self).__init__(channel, reduction)
        self.conv1 = nn.Conv3d(
            in_channels=self.channel, out_channels=self.channel // self.reduction, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(num_features=self.channel // self.reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv3d(
            in_channels=self.channel // self.reduction, out_channels=self.channel, kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(num_features=self.channel)

    def forward(self, x):
        b, c, t, _, _ = x.size()  # n, c, t, h, w
        y = self.conv1(x)  # n, c/r, t, h, w
        y = self.bn1(y)  # n, c/r, t, h, w
        y = self.avg_pool(y)  # n, c/r, 1, 1, 1
        y = self.conv2(y)  # n, c, 1, 1, 1
        y = self.bn2(y)  # n, c, 1, 1, 1
        y = self.sigmoid(y)  # n, c, 1, 1, 1
        # out = x * y.expand_as(x)  # n, c, t, h, w
        y = y - 0.5
        out = x + x * y.expand_as(x)
        return out


class SELayerMC(SELayer):
    """Construct channel-wise SELayer with max pooling."""

    def __init__(self, channel, reduction=16):
        super(SELayerMC, self).__init__(channel, reduction)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.max_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        # out = x * y.expand_as(x)
        y = y - 0.5
        out = x + x * y.expand_as(x)
        return out


class SELayerMAC(SELayer):
    """Construct channel-wise SELayer with the mix of average pooling and max pooling."""

    def __init__(self, channel, reduction=16):
        super(SELayerMAC, self).__init__(channel, reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 2), bias=False)
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)
        y = torch.cat((y_avg, y_max), dim=2).squeeze().unsqueeze(dim=1)
        y = self.conv(y).squeeze()
        y = self.fc(y).view(b, c, 1, 1, 1)
        # out = x * y.expand_as(x)
        y = y - 0.5
        out = x + x * y.expand_as(x)
        return out


