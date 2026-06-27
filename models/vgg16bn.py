# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

if __name__ == '__main__':
    from utils import UpSample_P2P, SimpleDecoder, TemporalUnit
else:
    from .utils import UpSample_P2P, SimpleDecoder, TemporalUnit

class VGG16_BN(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        vgg = models.vgg16_bn(pretrained=True)
        features = list(vgg.features.children())
        lids = [0, 33, 43]
        self.encoders = nn.ModuleList(nn.Sequential(*features[a:b]) for a, b in zip(lids[:-1], lids[1:]))
        self.num_channels = [512, 512]
        self.num_stage = len(self.num_channels)
        self.fuse_layer = UpSample_P2P(self.num_channels, ouc=256, bn=False, relu=False)
        
        # decoder (existing)
        self.decoders = SimpleDecoder(
            in_channel = self.fuse_layer.fuse_channel, 
            fea_channel = self.fuse_layer.fuse_channel, 
            up_scale = 2, 
            out_channel = 1   # ensure decoder outputs 1 channel density map
        )

        # temporal module configuration (can be moved to config)
        # temporal_in_channels should match decoder out_channel (here 1)
        self.temporal_in_channels = 1
        self.temporal_hidden = 32 if config is None or not hasattr(config, 'temporal_hidden') else config.temporal_hidden
        self.temporal_kernel = 3 if config is None or not hasattr(config, 'temporal_kernel') else config.temporal_kernel
        # small projection in case decoder out channels differ
        self.to_temporal = nn.Conv2d(self.temporal_in_channels, self.temporal_in_channels, kernel_size=1)
        self.temporal = TemporalUnit(mode='convgru', in_ch=self.temporal_in_channels, hid_ch=self.temporal_hidden, kernel_size=self.temporal_kernel)
        # map hidden -> logits (1 channel)
        self.hidden2logit = nn.Conv2d(self.temporal_hidden, 1, kernel_size=1)

        # store down factor hint (decoder / encoder ratio). Keep for P2R down argument if needed.
        # Set default down; adjust in training script as appropriate
        self.down = 4

    def forward(self, image, prev_h=None, need_fp=False):
        """
        Forward for single frame.
        Args:
            image: (B,C,H,W)
            prev_h: None or (B, temporal_hidden, H_out, W_out)
            need_fp: same as original flag
        Returns:
            pred_logits: (B,1,H_out,W_out)
            next_h: (B,temporal_hidden,H_out,W_out)
        """
        fea2 = self.encoding(image)
        if need_fp:
            fea2 = F.dropout2d(fea2, p=0.5)
        denmap = self.decoding(fea2)  # expected shape (B,1,H_out,W_out)
        # ensure channel compatibility
        x_t = self.to_temporal(denmap)
        next_h = self.temporal(x_t, prev_h)
        pred_logits = self.hidden2logit(next_h)
        return pred_logits, next_h

    def encoding(self, x):
        feas = []
        for module in self.encoders:
            feas.append(x := module(x))
        feas = feas[-self.num_stage:]
        fea = self.fuse_layer(feas)
        return fea

    def decoding(self, fea2):
        denmap = self.decoders(fea2)
        # if decoder produced >1 channels (not expected now), compute diff logic same as original
        if denmap.size(1) > 1:
            den1, den2 = denmap[:, :1], denmap[:, 1:2]
            den = den2 - den1
        else:
            den = denmap
        return den

    def init_hidden(self, batch_size, device=None, spatial_size=None, dtype=torch.float32):
        """
        Initialize hidden state for ConvGRU.
        spatial_size: (H_out, W_out) of decoder output.
        If spatial_size is None, user must have called forward once or set an attribute _temporal_spatial_size.
        """
        if device is None:
            device = next(self.parameters()).device
        if spatial_size is None:
            if hasattr(self, "_temporal_spatial_size"):
                H, W = self._temporal_spatial_size
            else:
                raise ValueError("spatial_size must be provided on first call to init_hidden")
        else:
            H, W = spatial_size
            self._temporal_spatial_size = (H, W)
        return torch.zeros(batch_size, self.temporal_hidden, H, W, device=device, dtype=dtype)


if __name__ == '__main__':
    model = VGG16_BN(None).cuda()
    x = torch.randn(1, 3, 256, 256).cuda()
    y, h = model(x)
    print(x.shape, y.shape, h.shape)
