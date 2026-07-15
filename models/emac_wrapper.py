# -*- coding: utf-8 -*-

import sys
import os
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial

EMAC_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'E-MAC'))
if EMAC_PATH not in sys.path:
    sys.path.insert(0, EMAC_PATH)

# ── Pure PyTorch Correlation fallback for PWC-Net ──────────────────────────
# Replaces the un-compilable correlation_cuda CUDA extension with a native
# PyTorch implementation using unfold + einsum.

class _CorrelationPyTorch(nn.Module):
    """Pure-PyTorch correlation layer (cost volume).
    For each position in f1, compute dot-product similarity with a
    (2*max_displacement+1)² neighborhood in f2.
    """
    def __init__(self, pad_size=4, kernel_size=1, max_displacement=4,
                 stride1=1, stride2=1, corr_multiply=1):
        super().__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, f1, f2):
        B, C, H, W = f1.shape
        D = self.max_displacement
        kW = 2 * D + 1

        # Pad f2 by D to keep output spatial size = H × W
        f2_unfold = F.unfold(f2, kernel_size=(kW, kW), padding=D,
                             stride=self.stride2)                     # (B, C*kW², H*W)
        f2_unfold = f2_unfold.view(B, C, kW * kW, H * W)             # (B, C, kW², H*W)

        f1_flat = f1.view(B, C, H * W).unsqueeze(2)                  # (B, C, 1, H*W)
        corr = (f1_flat * f2_unfold).sum(dim=1)                      # (B, kW², H*W)
        if self.corr_multiply != 1:
            corr = corr / C
        return corr.view(B, kW * kW, H, W)


# Inject the PyTorch Correlation into the correlation_package module path
# so PWCNet's `from .correlation_package.correlation import Correlation`
# resolves to our pure-PyTorch version.
import importlib
_corr_mod = types.ModuleType('correlation_package._correlation')
_corr_mod.Correlation = _CorrelationPyTorch
sys.modules['emac.models.correlation_package.correlation'] = _corr_mod

# Also ensure correlation_cuda mock exists so correlation.py can import it
try:
    import correlation_cuda
except ImportError:
    _mock_cc = types.ModuleType('correlation_cuda')
    _mock_cc.__file__ = os.path.join(EMAC_PATH, 'emac', 'models',
                                     'correlation_package', '_mock_corr.py')
    _mock_cc.__package__ = 'correlation_cuda'
    _mock_cc.__spec__ = importlib.machinery.ModuleSpec(
        name='correlation_cuda', loader=None, origin=_mock_cc.__file__)
    sys.modules['correlation_cuda'] = _mock_cc

from emac.emac import EMac
from emac.input_adapters import PatchedInputAdapter
from emac.output_adapters import SpatialOutputAdapter
from emac.emac_utils import TransFuse, Block, trunc_normal_, warp, denormalize


def _render_gaussian_density(points, shape, sigma=4, down=1):
    """Convert point annotations to a coarse gaussian density map.

    points: list[B] of (N_i, 2) tensors in (x, y) at original image resolution
    shape: (H, W) target density map resolution
    sigma: gaussian kernel std in target resolution pixels
    down: downsample factor from original to target (e.g., down=4 for 1/4 res)
    returns: (B, 1, H, W) float density map
    """
    B = len(points)
    H, W = shape
    device = points[0].device if B > 0 else 'cpu'
    device = torch.device(device)
    density = torch.zeros(B, 1, H, W, device=device)

    sigma = max(sigma, 1.0)
    ks = int(sigma * 3) * 2 + 1
    ax = torch.arange(-ks // 2 + 1, ks // 2 + 1, device=device).float()
    xx, yy = torch.meshgrid(ax, ax, indexing='xy')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()

    for b in range(B):
        pts = points[b]
        if pts.numel() == 0:
            continue
        xs = (pts[:, 0] / down).round().long()
        ys = (pts[:, 1] / down).round().long()
        valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        xs, ys = xs[valid], ys[valid]
        if xs.numel() == 0:
            continue
        uniq = torch.stack([xs, ys], dim=1).unique(dim=0)
        density[b, 0, uniq[:, 1], uniq[:, 0]] = 1.0

    if sigma > 0:
        pad = ks // 2
        density = F.pad(density, (pad, pad, pad, pad), mode='replicate')
        density = F.conv2d(density, kernel[None, None, :, :], padding=0)
        density = density[:, :, :H, :W]

    return density


class EMACWrapper(nn.Module):
    """Wrapper adapting native E-MAC to P2RLoss interface.

    Forward interface (per user spec):
        pred_density = model(img_current, templates=[img_template], flow=current_flow)
    """

    def __init__(self, config=None):
        super().__init__()

        input_size = 256
        patch_size = 16
        num_encoded_tokens = 128
        total_num_tokens = 256
        decoder_depth = 2
        decoder_dim = 512
        decoder_num_heads = 16
        num_global_tokens = 1
        drop_path = 0.3

        if config is not None:
            input_size = getattr(config, 'EMAC_INPUT_SIZE', input_size)
            patch_size = getattr(config, 'EMAC_PATCH_SIZE', patch_size)
            num_encoded_tokens = getattr(config, 'EMAC_NUM_ENCODED_TOKENS', num_encoded_tokens)
            total_num_tokens = getattr(config, 'EMAC_TOTAL_NUM_TOKENS', total_num_tokens)
            decoder_depth = getattr(config, 'EMAC_DECODER_DEPTH', decoder_depth)
            decoder_dim = getattr(config, 'EMAC_DECODER_DIM', decoder_dim)
            decoder_num_heads = getattr(config, 'EMAC_DECODER_NUM_HEADS', decoder_num_heads)
            num_global_tokens = getattr(config, 'EMAC_NUM_GLOBAL_TOKENS', num_global_tokens)
            drop_path = getattr(config, 'EMAC_DROP_PATH', drop_path)
            self.density_sigma = getattr(config, 'EMAC_DENSITY_SIGMA', 4)
        else:
            self.density_sigma = 4

        self.input_size = input_size
        self.patch_size = patch_size
        self.num_encoded_tokens = num_encoded_tokens
        self.total_num_tokens = total_num_tokens
        self.down = 4

        input_adapters = {
            "rgb": PatchedInputAdapter(
                num_channels=3, stride_level=1,
                patch_size_full=patch_size, image_size=input_size,
            ),
            "density": PatchedInputAdapter(
                num_channels=1, stride_level=1,
                patch_size_full=patch_size, image_size=input_size,
            ),
        }

        output_adapters = {
            "density": SpatialOutputAdapter(
                stride_level=1, patch_size_full=patch_size,
                image_size=input_size, num_channels=1,
                dim_tokens_enc=768, dim_tokens=decoder_dim,
                depth=decoder_depth, use_task_queries=True,
                task="density", context_tasks=["rgb", "density"],
            ),
        }

        fuse_module = TransFuse(
            stride_level=1, patch_size_full=patch_size,
            image_size=input_size, num_channels=1,
            dim_tokens_enc=768, dim_tokens=decoder_dim,
            num_heads=decoder_num_heads,
        )

        self.emac = EMac(
            input_adapters=input_adapters,
            output_adapters=output_adapters,
            fuse_module=fuse_module,
            dim_tokens=768, depth=12, num_heads=12, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_global_tokens=num_global_tokens, drop_path_rate=drop_path,
        )

        self._input_adapters = input_adapters
        self._output_adapters = output_adapters
        self._fuse = fuse_module
        self._encoder = self.emac.encoder

    @property
    def pwc(self):
        """Access the built-in PWC-Net optical flow network."""
        return self.emac.pwc

    def compute_flow(self, img_prev, img_cur):
        """End-to-end PWC-Net flow from prev→cur, upscaled to full resolution.

        Args:
            img_prev, img_cur: (B, 3, H, W) in ImageNet-normalized space
        Returns:
            flo: (B, 2, H, W) flow at original resolution, upscaled from native PWC
        """
        B, C, H, W = img_cur.shape
        flo_raw = self._compute_flow_raw(img_prev, img_cur)          # (B, 2, H//4, W//4)
        flo = F.interpolate(flo_raw, (H, W), mode='bilinear', align_corners=False)
        flo = flo * (flo_raw.shape[-2] * flo_raw.shape[-1]) / (H * W)
        return flo

    def _compute_flow_raw(self, img_prev, img_cur):
        """PWC-Net flow at native resolution (H_img//4 × W_img//4).

        Returns:
            flo_raw: (B, 2, H//4, W//4) — no bilinear upscaling
        """
        permute_bgr = [2, 1, 0]
        img_all = torch.cat([
            denormalize(img_cur)[:, permute_bgr] / 255.0,
            denormalize(img_prev)[:, permute_bgr] / 255.0,
        ], dim=1).to(img_cur.device)
        return self.emac.pwc(img_all)

    def forward(self, img_current, templates=None, flow=None, density_ref=None, task_masks=None,
                return_aux=False):
        """Forward pass.

        When ``flow`` is provided (precomputed, e.g. RAFT), uses it for
        temporal fusion.  When ``flow`` is None, falls back to the built-in
        PWC-Net.

        Args:
            img_current: (B, 3, H, W) current frame
            templates: list of one tensor (B, 3, H, W) — previous / template frame
            flow: (B, 2, H, W) optional precomputed optical flow
            density_ref: (B, 1, H, W) coarse density for masking guidance
            task_masks: optional precomputed masks dict
            return_aux: if True, returns (pred_fuse, img_warp, pred_prev_warp, pred_cur)
        """
        img_template = templates[0] if templates else img_current
        B, C, H, W = img_current.shape

        x_dict = {
            "rgb": torch.stack([img_template, img_current], dim=2),
            "density": density_ref.unsqueeze(2).expand(-1, -1, 2, -1, -1).contiguous()
                       if density_ref is not None
                       else torch.zeros(B, 1, 2, H, W, device=img_current.device),
        }

        # E-MAC core logic (without PWCNet); run in fp32 for stability
        B = x_dict["rgb"].shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            inp_cur = {
                d: self._input_adapters[d](x_dict[d][:, :, -1].float())
                for d in self._input_adapters if d in x_dict
            }
            inp_prev = {
                d: self._input_adapters[d](x_dict[d][:, :, 0].float())
                for d in self._input_adapters if d in x_dict
            }
        input_info = self.emac.generate_input_info(inp_cur, (H, W))
        input_info_prev = self.emac.generate_input_info(inp_prev, (H, W))

        num_tokens = sum(t.shape[1] for t in inp_cur.values())

        if task_masks is None:
            import random as _random
            _rand = torch.rand(B, num_tokens, device=img_current.device)
            ids_shuffle = torch.argsort(_rand, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            n_keep = min(self.num_encoded_tokens, num_tokens)
            ids_keep = ids_shuffle[:, :n_keep]
            ids_keep_prev = ids_keep
            ids_restore_prev = ids_restore
        else:
            mask_all = torch.cat([task_masks[t] for t in inp_cur.keys()], dim=1)
            ids_shuffle = torch.argsort(mask_all, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :self.num_encoded_tokens]
            ids_restore_prev = ids_restore
            ids_keep_prev = ids_keep

        tokens_cur = torch.cat(list(inp_cur.values()), dim=1)
        tokens_cur = torch.gather(tokens_cur, 1, ids_keep.unsqueeze(-1).repeat(1, 1, tokens_cur.shape[2]))
        tokens_prev = torch.cat(list(inp_prev.values()), dim=1)
        tokens_prev = torch.gather(tokens_prev, 1, ids_keep_prev.unsqueeze(-1).repeat(1, 1, tokens_prev.shape[2]))

        global_tok = self.emac.global_tokens.expand(B, -1, -1)
        tokens_cur = torch.cat([tokens_cur, global_tok], dim=1)
        tokens_prev = torch.cat([tokens_prev, global_tok], dim=1)

        with torch.cuda.amp.autocast(enabled=False):
            enc_cur = self._encoder(tokens_cur)
            enc_prev = self._encoder(tokens_prev)

            pred_cur = self._output_adapters["density"](
                encoder_tokens=enc_cur, input_info=input_info,
                ids_keep=ids_keep, ids_restore=ids_restore
            )
            pred_prev = self._output_adapters["density"](
                encoder_tokens=enc_prev, input_info=input_info_prev,
                ids_keep=ids_keep_prev, ids_restore=ids_restore_prev
            )

        # Temporal fusion: low-res warp preserves micro-motion, then upsample warped density
        with torch.cuda.amp.autocast(enabled=False):
            if flow is not None:
                flow_resized = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
                scale = torch.tensor([flow.shape[-1] / W, flow.shape[-2] / H], device=flow.device)
                flow_resized = flow_resized * scale.view(1, 2, 1, 1)
                pred_prev_warp = warp(pred_prev, flow_resized.detach())
                flo = None
            else:
                # Warp at native PWC resolution (H//4, W//4) to avoid flow over-smoothing
                flo_raw = self._compute_flow_raw(img_template, img_current)     # (B,2,H/4,W/4)
                pred_prev_low = F.avg_pool2d(pred_prev, 4, 4)                   # (B,1,H/4,W/4)
                pred_prev_warp_low = warp(pred_prev_low, flo_raw.detach())
                pred_prev_warp = F.interpolate(pred_prev_warp_low, (H, W),
                                               mode='bilinear', align_corners=False)
                # Keep upscaled flow for opt_loss / TV loss in train loop
                flo = F.interpolate(flo_raw, (H, W), mode='bilinear', align_corners=False)
                flo = flo * (flo_raw.shape[-2] * flo_raw.shape[-1]) / (H * W)

            pred_fuse = self._fuse_dense(pred_prev_warp, pred_cur)

        if return_aux:
            return pred_fuse, flo, pred_prev_warp, pred_cur
        return pred_fuse

    def _fuse_dense(self, pred_prev_warp, pred_cur):
        tok_warp = self._input_adapters["density"](pred_prev_warp.float())
        tok_cur = self._input_adapters["density"](pred_cur.float())
        return pred_cur + self._fuse(tok_warp, tok_cur)

    def init_hidden(self, batch_size, device=None, spatial_size=None):
        return None

    def prepare_density_ref(self, gt_points, img_shape):
        """Convert point annotations to gaussian density for E-MAC masking guidance.

        gt_points: list[B] of (N_i, 2) tensors at original image resolution
        img_shape: (H, W) of the input image
        returns: (B, 1, H, W) density map at full resolution
        """
        return _render_gaussian_density(
            gt_points, img_shape, sigma=self.density_sigma, down=1
        )
