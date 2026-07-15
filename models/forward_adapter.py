# -*- coding: utf-8 -*-
"""
models/forward_adapter.py

Uniform forward interface for video (clip-level) models, abstracting away
the difference between recurrent BPTT (VGG16BN), 2-frame sliding-window
(EMAC), and future clip-level transformers.

Usage::

    adapter = ModelForwardAdapter(model, model_name)
    logits = adapter.forward_video(clip, templates=None, flow=None)
    #  clip:      (B, T, C, H, W)
    #  templates:  optional list of (B, C, H, W)
    #  flow:       optional (B, 2, H, W)
    #  returns:    (B, 1, H_out, W_out)
"""

import torch
import torch.nn as nn


class ModelForwardAdapter:
    """Wraps different model forward conventions behind a single interface.

    The adapter does **not** modify the underlying model; it only provides
    a unified ``forward_video`` method that hides per-model boilerplate.
    """

    def __init__(self, model: nn.Module, model_name: str = 'vgg16bn'):
        self.model = model
        self.model_name = model_name.lower()

    @property
    def is_recurrent(self) -> bool:
        """True if the model expects ``(image, prev_h)`` and returns
        ``(logits, next_h)``."""
        return self.model_name == 'vgg16bn'

    def forward_video(
        self,
        clip: torch.Tensor,
        templates=None,
        flow=None,
        density_ref=None,
        return_aux=False,
    ):
        """Run model over a video clip.

        Args:
            clip:        (B, T, C, H, W) full clip
            templates:   optional list of one (B, C, H, W) tensor (EMAC-style)
            flow:        optional (B, 2, H, W) precomputed optical flow
            density_ref: optional (B, 1, H, W) coarse density for masking
            return_aux:  if True, additionally returns intermediate tensors

        Returns:
            pred_fuse: (B, 1, H, W)

            If ``return_aux`` is True, returns a tuple
            ``(pred_fuse, flo, pred_prev_warp, pred_cur)``.
        """
        if self.is_recurrent:
            return self._forward_recurrent(clip)
        else:
            return self._forward_clip(clip, templates, flow, density_ref, return_aux)

    def _forward_recurrent(self, clip: torch.Tensor):
        """BPTT-style: single-frame forward with hidden state propagation.

        Only the **last frame** prediction is returned.
        """
        B, T, C, H, W = clip.shape
        prev_h = None
        last_logits = None
        for t in range(T):
            frame = clip[:, t]
            last_logits, prev_h = self.model(frame, prev_h=prev_h)
        return last_logits

    def _forward_clip(self, clip, templates, flow, density_ref, return_aux):
        """Clip-level: process at most 2 frames with optional templates/flow."""
        B, T, C, H, W = clip.shape
        cur = clip[:, 0]
        if templates is not None:
            pass  # use caller-provided templates
        elif T >= 2:
            templates = [clip[:, 1]]
        else:
            templates = [cur]

        if return_aux:
            return self.model(
                cur, templates=templates, flow=flow,
                density_ref=density_ref, return_aux=True
            )
        return self.model(
            cur, templates=templates, flow=flow,
            density_ref=density_ref
        )
