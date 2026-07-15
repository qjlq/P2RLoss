# -*- coding: utf-8 -*-

import torch
from .vgg16bn import VGG16_BN
from .emac_wrapper import EMACWrapper


def build_model(config):
    """Build student & teacher models.

    Args:
        config: The full YACS config object (root level, not just MODEL).
    """
    model_name = config.MODEL.NAME.lower()
    model_cls = {
        'vgg16bn': VGG16_BN,
        'emac': EMACWrapper,
    }[model_name]

    if model_name == 'emac':
        student = model_cls(config)
        teacher = model_cls(config)
    else:
        student = model_cls(config)
        teacher = model_cls(config)

    # ── self.down sanity check ─────────────────────────────────────────────
    down = getattr(student, 'down', None)
    if down is not None:
        _check_down_factor(student, down)
    else:
        import warnings
        warnings.warn(
            f"Model {config.NAME} does not define self.down; "
            "P2R loss scaling will use a fallback value of 4."
        )

    return student, teacher


def _check_down_factor(model, expected_down, test_size=256):
    """Run a single forward pass and verify output spatial size.

    Warns if ``output_H * expected_down != test_size`` — this can indicate
    a misconfigured ``self.down``.
    """
    device = next(model.parameters()).device
    dummy = torch.zeros(1, 3, test_size, test_size, device=device)
    model.eval()
    with torch.no_grad():
        try:
            out, _ = model(dummy, prev_h=None)
        except TypeError:
            out = model(dummy)
        except Exception:
            return

    actual_down = test_size // out.shape[-1]
    computed = out.shape[-1] * expected_down
    if computed != test_size:
        import warnings
        warnings.warn(
            f"{type(model).__name__}: self.down={expected_down} but output "
            f"spatial size = {out.shape[-2]}×{out.shape[-1]} on a {test_size}×{test_size} "
            f"input (ratio = {actual_down}). "
            f"P2R loss will map density pixels to ({computed}×{computed}) space while "
            f"GT points are in ({test_size}×{test_size}) space. "
            f"Ensure this is intentional."
        )
    model.train()
