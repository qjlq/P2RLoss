# -*- coding: utf-8 -*-

import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as tF

def convblock(inc, ouc, kernel_size, bn=True):
    padding = kernel_size // 2
    module = nn.Sequential(
        nn.Conv2d(inc, ouc, kernel_size=kernel_size, stride=1, padding=padding, bias=not bn),
        nn.BatchNorm2d(ouc) if bn else nn.Identity(),
        nn.ReLU(inplace=True)
    )
    if not bn: 
    #     init.kaiming_uniform_(module[0].weight, mode='fan_in', nonlinearity='relu')
        init.constant_(module[0].bias, 0.)

    return module


def conv_3x3(inc, ouc, bn=True):
    return convblock(inc, ouc, kernel_size=3, bn=bn)

class UpSample_P2P(nn.Module):
    def __init__(self, incs, ouc, bn=True, relu=True):
        super().__init__()
        self.align_layers = nn.ModuleList([
            nn.Conv2d(inc, ouc, kernel_size=1, stride=1, padding=0, bias=not bn) for inc in incs
        ])
        if not bn:
            for layer in self.align_layers:
                init.constant_(layer.bias, 0.)

        self.fuse = nn.Sequential(
            nn.Conv2d(ouc, ouc, kernel_size=3, stride=1, padding=1, bias= not bn),
            nn.BatchNorm2d(ouc) if bn else nn.Identity(),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

        self.fuse_channel = ouc
    
    def forward(self, xs):
        x0 = self.align_layers[0](xs[0])
        out_shape = x0.shape[-2:]
        for x, layer in zip(xs[1:], self.align_layers[1:]):
            x = tF.interpolate(layer(x), out_shape, mode='bilinear', align_corners=False)
            x0 = x0 + x
        x = self.fuse(x0)
        return x



class SimpleDecoder(nn.Sequential):
    def __init__(self, in_channel = 128, fea_channel=64, up_scale=1, out_channel=1):
        super().__init__(
            conv_3x3(in_channel, fea_channel, bn=False),
            conv_3x3(fea_channel, fea_channel, bn=False),
            nn.Conv2d(fea_channel, out_channel * (up_scale ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(up_scale)
        )
        self.up_scale = up_scale
        init.constant_(self[-2].bias, 0.)


# --- start add ConvGRU / TemporalUnit ---
class ConvGRUCell(nn.Module):
    """
    Lightweight ConvGRU cell.
    input: x (B, in_ch, H, W), h_prev (B, hid_ch, H, W)
    return: h_next (B, hid_ch, H, W)
    """
    def __init__(self, in_ch, hid_ch, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.in_ch = in_ch
        self.hid_ch = hid_ch
        # gates: z and r
        self.conv_zr = nn.Conv2d(in_ch + hid_ch, 2 * hid_ch, kernel_size=kernel_size, padding=padding)
        # candidate
        self.conv_n = nn.Conv2d(in_ch + hid_ch, hid_ch, kernel_size=kernel_size, padding=padding)
        # initialization
        init.kaiming_normal_(self.conv_zr.weight, nonlinearity='relu')
        init.constant_(self.conv_zr.bias, 0.)
        init.kaiming_normal_(self.conv_n.weight, nonlinearity='relu')
        init.constant_(self.conv_n.bias, 0.)

    def forward(self, x, h):
        # x: (B, in_ch, H, W)
        # h: (B, hid_ch, H, W)
        if h is None:
            # init zeros if needed
            h = torch.zeros(x.size(0), self.hid_ch, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        cat = torch.cat([x, h], dim=1)
        zr = self.conv_zr(cat)
        z, r = torch.sigmoid(zr.chunk(2, dim=1))
        cat_r = torch.cat([x, r * h], dim=1)
        n = torch.tanh(self.conv_n(cat_r))
        h_next = (1 - z) * n + z * h
        return h_next

class TemporalUnit(nn.Module):
    """
    Simple wrapper to select temporal mode. Currently implements ConvGRU.
    mode: 'convgru' (default). TSM could be added here if desired.
    """
    def __init__(self, mode='convgru', in_ch=1, hid_ch=32, kernel_size=3):
        super().__init__()
        assert mode in ('convgru', 'tsm'), "mode must be 'convgru' or 'tsm'"
        self.mode = mode
        if mode == 'convgru':
            self.cell = ConvGRUCell(in_ch, hid_ch, kernel_size=kernel_size)
        else:
            # placeholder for TSM or other temporal modules
            raise NotImplementedError("TSM mode not implemented in this wrapper yet")

    def forward(self, x, h):
        # x: (B, in_ch, H, W); h: (B, hid_ch, H, W) or None
        return self.cell(x, h)
# --- end add ConvGRU / TemporalUnit ---
