# -*- coding: utf-8 -*-
"""
train/train_loop.py

Core training loop with temporal truncated BPTT, teacher-student EMA and temporal P2R calibration.

Expectations:
- student and teacher models implement forward(image, prev_h=None) -> (pred_logits, next_h)
- models provide init_hidden(batch_size, device, spatial_size)
- Dataset returns sequences; unlabeled batch contains flows as precomputed tensors with shape (B, T-1, 2, H, W) or None
- P2RLoss is available at losses.p2rloss.P2RLoss

This module provides train_one_epoch implementing the requested logic.

"""

import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from losses.p2rloss import P2RLoss
from tqdm import tqdm
import logging
from models.forward_adapter import ModelForwardAdapter

# Import E-MAC utilities for built-in flow computation (only used by EMAC branch)
try:
    from emac.emac_utils import denormalize, warp as emac_warp
except ImportError:
    def denormalize(x, mean=None, std=None):
        if mean is None:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        import torchvision.transforms.functional as TF
        return TF.normalize(x.clone(),
                            mean=[-m / s for m, s in zip(mean, std)],
                            std=[1 / s for s in std])

    def emac_warp(x, flo):
        B, C, H, W = x.shape
        xx = torch.arange(W, device=x.device).view(1, -1).repeat(H, 1)
        yy = torch.arange(H, device=x.device).view(-1, 1).repeat(1, W)
        grid = torch.stack([xx, yy], dim=0).float().unsqueeze(0).repeat(B, 1, 1, 1)
        vgrid = grid + flo
        vgrid[:, 0] = 2.0 * vgrid[:, 0] / max(W - 1, 1) - 1.0
        vgrid[:, 1] = 2.0 * vgrid[:, 1] / max(H - 1, 1) - 1.0
        return F.grid_sample(x, vgrid.permute(0, 2, 3, 1), mode='bilinear', align_corners=True)


def tv_loss(x):
    """Total variation smoothness loss."""
    return (F.l1_loss(x[:, :, :-1, :], x[:, :, 1:, :]) +
            F.l1_loss(x[:, :, :, :-1], x[:, :, :, 1:])) / x.size(1)


def update_ema(student, teacher, alpha):
    """Exponential moving average of parameters: teacher = alpha * teacher + (1-alpha) * student"""
    for sp, tp in zip(student.parameters(), teacher.parameters()):
        tp.data.mul_(alpha).add_(sp.data, alpha=1.0 - alpha)


def generate_points_from_density(density_logits, threshold=0.3, topk=None):
    """
    Extract point coordinates from a density/logits map using local-max and thresholding.

    Resolution-agnostic: works with ANY (H, W) input — no hardcoded stride/scale.
    Coordinates are returned at the SAME resolution as the input density map.

    Args:
        density_logits: Tensor (B, 1, H, W) logits (before sigmoid)
        threshold:      sigmoid probability threshold for peak detection
        topk:           if set, keep only top-k highest-probability points
    Returns:
        list_of_points: length B list, each is Tensor (N_i, 2) in (x, y)
    """
    prob = torch.sigmoid(density_logits)
    B, C, H, W = prob.shape
    assert C == 1, f"Expected 1 channel, got {C}"
    # ── Diagnostic: confirm input resolution ────────────────────────────────
    # (will be removed after debug, kept as permanent guard)
    assert H > 0 and W > 0, f"Empty density map: {H}×{W}"

    # Local-max NMS — kernel=3, stride=1, padding=1 preserves spatial size
    pool = F.max_pool2d(prob, kernel_size=3, stride=1, padding=1)
    peaks = (prob == pool) & (prob > threshold)

    points_list = []
    for b in range(B):
        ys, xs = torch.nonzero(peaks[b, 0], as_tuple=True)
        if xs.numel() == 0:
            points_list.append(
                torch.zeros((0, 2), device=density_logits.device, dtype=torch.float32)
            )
            continue
        # (row, col) → (x, y) : column → x, row → y
        coords = torch.stack([xs.float(), ys.float()], dim=1)

        # ── Guard: coordinates MUST match input resolution ─────────────────
        x_max, y_max = coords[:, 0].max().item(), coords[:, 1].max().item()
        assert x_max < W, f"x={x_max} exceeds width={W}"
        assert y_max < H, f"y={y_max} exceeds height={H}"
        # Warn if coordinates don't span the full range (suggests resolution mismatch)
        if x_max < W * 0.5 or y_max < H * 0.5:
            import warnings as _w
            _w.warn(
                f"generate_points_from_density: coordinates only span "
                f"[0, {x_max:.0f}]×[0, {y_max:.0f}] for a {H}×{W} input "
                f"(>50% of spatial range empty). This may indicate a "
                f"resolution mismatch upstream."
            )

        if topk is not None and coords.shape[0] > topk:
            vals = prob[b, 0, ys, xs]
            _, idx = vals.topk(topk)
            coords = coords[idx]
        points_list.append(coords)

    return points_list


def sample_flow_at_points(flow, points, flow_down=2, model_down=4):
    """
    Sample per-point flow vectors from a dense flow map using grid_sample.
    Args:
        flow: Tensor (B, 2, H, W)
        points: Tensor (B, N, 2) in (x, y) pixel coordinates at model density-map resolution.
                If N==0 returns empty tensor.
        flow_down: downsampling factor used when computing flow (e.g. 2)
        model_down: downsampling factor of the model's density map (e.g. 4)
    Returns:
        sampled: Tensor (B, N, 2) flow vectors (dx, dy) in flow-map pixels
    """
    B, _, H, W = flow.shape
    device = flow.device
    scale = model_down / flow_down  # density-map → flow-map coordinate scale
    sampled_list = []
    for b in range(B):
        pts = points[b]  # (N,2)
        if pts.numel() == 0:
            sampled_list.append(torch.zeros((0, 2), device=device))
            continue
        xs = pts[:, 0] * scale
        ys = pts[:, 1] * scale
        # normalize to [-1,1] in flow-map space
        nx = (xs / float(max(W - 1, 1))) * 2.0 - 1.0
        ny = (ys / float(max(H - 1, 1))) * 2.0 - 1.0
        grid = torch.stack((nx, ny), dim=1).unsqueeze(0).unsqueeze(0)  # (1,1,N,2)
        f = flow[b:b+1]
        sampled = F.grid_sample(f, grid.view(1, -1, 1, 2), align_corners=True).view(2, -1).permute(1, 0)  # (N,2)
        sampled_list.append(sampled)
    return sampled_list


def warp_points_with_flow(points_list, flow_batch, flow_down=2, model_down=4):
    """
    Warp a list of points (per-batch) using flow maps for corresponding frame-pairs.
    Points are at model density-map resolution; flow is at flow-map resolution.
    Args:
        points_list: list length B of tensors (N_i,2) in (x,y) at model density-map resolution
        flow_batch: Tensor (B, 2, H, W) flow from prev->curr (dx,dy) in flow-map pixels
        flow_down: downsampling factor used when computing flow (e.g. 2)
        model_down: downsampling factor of the model's density map (e.g. 4)
    Returns:
        warped_list: list length B of tensors (N_i,2) warped to current frame, at model density-map resolution
    """
    B = flow_batch.shape[0]
    device = flow_batch.device
    scale = model_down / flow_down       # density→flow
    inv_scale = flow_down / model_down   # flow→density
    maxN = max([p.shape[0] for p in points_list]) if len(points_list) > 0 else 0
    if maxN == 0:
        return [torch.zeros((0, 2), device=device) for _ in range(B)]
    pts_padded = torch.zeros((B, maxN, 2), device=device)
    for b in range(B):
        n = points_list[b].shape[0]
        if n > 0:
            pts_padded[b, :n] = points_list[b].to(device)
    sampled_list = sample_flow_at_points(flow_batch, pts_padded, flow_down, model_down)
    warped = []
    for b in range(B):
        if points_list[b].shape[0] == 0:
            warped.append(points_list[b].new_zeros((0, 2)))
            continue
        pts_flow = points_list[b].to(device) * scale          # to flow-map space
        flow_vecs = sampled_list[b].to(device)                # flow displacement in flow-map pixels
        warped_flow = pts_flow + flow_vecs                    # warp in flow-map space
        warped_pts = warped_flow * inv_scale                  # back to density-map space
        warped.append(warped_pts)
    return warped


def prepare_p2r_seqs_from_points_list(points_list):
    """
    Convert list of per-batch (N_i,2) into list of sequences expected by P2RLoss
    P2RLoss expects seqs as a list where each element corresponds to sample i and is a Tensor (Ni,2)
    So we simply return the points_list as-is (ensuring float and device consistency)
    """
    return [p.float() for p in points_list]


def get_adaptive_pos_weight(current_epoch, stage1_epochs, total_epochs):
    """Dynamic pos_weight decay across stages.

    STAGE_1 (cur < stage1_epochs): fixed at 20.0
    STAGE_2 first 40%: linear decay 20.0 → 2.0
    STAGE_2 last 60%: fixed at 2.0
    """
    if total_epochs <= stage1_epochs:
        return 20.0

    if current_epoch < stage1_epochs:
        return 20.0

    stage2_epochs = total_epochs - stage1_epochs
    tau = (current_epoch - stage1_epochs) / stage2_epochs
    decay_end = 0.4

    if tau < decay_end:
        alpha = tau / decay_end
        return 20.0 * (1.0 - alpha) + 2.0 * alpha
    else:
        return 2.0


def check_min_batch(batch_tensor, min_bs=2):
    if batch_tensor.size(0) < min_bs:
        raise ValueError(f"Batch size must be >= {min_bs} to avoid OOM; got {batch_tensor.size(0)}")


def train_one_epoch(
    epoch,
    dataloader,
    student,
    teacher,
    optimizer,
    device,
    down=4,
    ema_alpha=0.999,
    scaler=None,
    p2r_loss_cfg=None,
    amp_enabled=True,
    min_batch_size=2,
    stage='sup',
    stage1_epochs=25,
    total_epochs=50,
    model_name='vgg16bn',
    accumulation_steps=1,
):
    """
    Core training loop implementing truncated BPTT and temporal P2R calibration.
    
    stage='sup': supervised training on labeled data only.
    stage='semi': semi-supervised with supervised loss + temporal pseudo-labeling.
    
    - dataloader yields tuples depending on Dataset.collate_fn. Expected training batch:
      (limg_batch [B,T,C,H,W], lseqs_list, lids, uimg_batch [B,T,C,H,W], umask_batch [B,T,1,H,W], uids, uflows_batch or None)
    - student/teacher forward: (pred_logits, next_h) = model(frame, prev_h)
    - p2r used to compute loss between student_pred_t and warped teacher pseudo-points (teacher_pred_{t-1} warped by flow[t-1])
    - Truncated BPTT: after backward+step for each step, detach hidden: h = h.detach()
    """
    student.train()
    teacher.eval()
    p2r = P2RLoss() if p2r_loss_cfg is None else P2RLoss(**p2r_loss_cfg)
    if scaler is None:
        scaler = GradScaler()

    logger = logging.getLogger(model_name.upper() if model_name else 'VGG16BN')
    num_batches = len(dataloader)
    current_pos_weight = get_adaptive_pos_weight(epoch, stage1_epochs, total_epochs)
    logger.info(f"[epoch {epoch}] adaptive pos_weight = {current_pos_weight:.2f}")
    pbar = tqdm(dataloader, desc=f'Epoch [{epoch}]', leave=False)
    
    epoch_loss = 0.0
    for batch_idx, batch in enumerate(pbar):
        # Unpack batch; support older dataset formats by trying variants
        if len(batch) >= 7:
            limg_batch, lseqs, lids, uimg_batch, umask_batch, uids, uflows_batch = batch
        elif len(batch) == 3:
            limg_batch, lseqs, lids = batch
            uimg_batch = None
            umask_batch = None
            uflows_batch = None
        else:
            raise RuntimeError("Unsupported batch format from dataloader")

        # Move to device
        if limg_batch is not None:
            limg_batch = limg_batch.to(device)
        if uimg_batch is not None:
            uimg_batch = uimg_batch.to(device)
        if umask_batch is not None:
            umask_batch = umask_batch.to(device)
        if uflows_batch is not None:
            uflows_batch = uflows_batch.to(device)

        B, T, C, H, W = uimg_batch.shape

        is_emac = (model_name == 'emac')

        # === Supervised loss on labeled first frame ===
        sup_loss = 0.0
        if limg_batch is not None and limg_batch.size(0) > 0:
            lframe = limg_batch[:, 0]
            ldots = [seq[0][0][:, :2].to(device) for seq in lseqs]
            with autocast(enabled=amp_enabled):
                if is_emac:
                    img_template = limg_batch[:, 1] if T > 1 else lframe
                    # Labeled data always uses clean input (no asymmetric noise)
                    density_ref = student.prepare_density_ref(ldots, img_shape=(H, W))
                    lpred = student(lframe, templates=[img_template], density_ref=density_ref)
                else:
                    lpred, _ = student(lframe, prev_h=None)
            sup_loss = p2r(lpred, ldots, down=down, pos_weight=current_pos_weight)

            # ── Assert: 校驗 down 與實際輸入/輸出解析度一致 ──────────────
            # 注意: 當輸出解析度與輸入相同（actual_ratio=1），
            # down 為虛擬縮放因子（如 EMAC），跳過此檢查。
            H_in, W_in = lframe.shape[-2], lframe.shape[-1]
            H_out, W_out = lpred.shape[-2], lpred.shape[-1]
            actual_h = H_in / H_out
            actual_w = W_in / W_out
            if abs(actual_h - 1) > 1e-4:
                # 僅在實際有下採樣時校驗（例如 VGG16BN: 256→64, ratio=4）
                assert abs(actual_h - down) < 1e-4 and abs(actual_w - down) < 1e-4, (
                    f"\n{'='*60}\n"
                    f"🔴 down 參數不匹配！\n"
                    f"  輸入 ({H_in}×{W_in}) → 輸出 ({H_out}×{W_out})\n"
                    f"  實際比例 ({actual_h:.2f}×, {actual_w:.2f}×)\n"
                    f"  傳入 P2R 的 down={down}\n"
                    f"  請修正模型 self.down 或傳入正確的 down 值\n"
                    f"{'='*60}"
                )
            # ─────────────────────────────────────────────────────────────

            # ── Debug: 校驗實際降採樣率 vs self.down ──────────────────────
            Hin, Win = lframe.shape[-2], lframe.shape[-1]
            Hout, Wout = lpred.shape[-2], lpred.shape[-1]
            actual_down_h = Hin / Hout
            actual_down_w = Win / Wout
            if abs(actual_down_h - 1) > 1e-4:
                # 僅在實際有下採樣時校驗（VGG16BN），EMAC 等全解析度輸出跳過
                assert abs(actual_down_h - down) < 1e-4 and abs(actual_down_w - down) < 1e-4, (
                f"\n{'='*60}\n"
                f"🔴 down 參數不匹配！\n"
                f"  模型: {model_name}\n"
                f"  輸入尺寸: ({Hin}, {Win})\n"
                f"  輸出尺寸: ({Hout}, {Wout})\n"
                f"  實際縮放: ({actual_down_h:.2f}x, {actual_down_w:.2f}x)\n"
                f"  設定 down: {down}x\n"
                f"  建議: 修正模型 __init__ 中 self.down = {actual_down_h:.0f}\n"
                f"{'='*60}"
            )

        if stage == 'sup':
            optimizer.zero_grad()
            if amp_enabled:
                scaler.scale(sup_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                sup_loss.backward()
                optimizer.step()
            batch_loss = sup_loss.item()

        elif stage == 'semi':
            if is_emac:
                check_min_batch(uimg_batch, min_bs=min_batch_size)
                batch_loss = 0.0

                # === Asymmetric augmentation (student gets noise, teacher stays clean) ===
                if epoch >= stage1_epochs:
                    uimg_batch_teacher = uimg_batch.clone()
                    noise = torch.randn_like(uimg_batch) * 0.05
                    uimg_batch = torch.clamp(uimg_batch + noise, 0.0, 1.0)
                else:
                    uimg_batch_teacher = uimg_batch

                with torch.no_grad():
                    teacher_pred_fuse = teacher(
                        uimg_batch_teacher[:, 0], templates=[uimg_batch_teacher[:, 1]]
                    )
                    teacher_pred_prev = teacher_pred_fuse.detach()

                for t in range(1, T):
                    optimizer.zero_grad()
                    frame_t = uimg_batch[:, t]
                    frame_t_teacher = uimg_batch_teacher[:, t]

                    # Built-in PWC-Net flow instead of external .npy
                    with autocast(enabled=amp_enabled):
                        student_pred_t, flo_t, pred_prev_warp_raw, pred_cur_raw = student(
                            frame_t, templates=[uimg_batch[:, (t + 1) % T]],
                            return_aux=True
                        )

                    with torch.no_grad():
                        pseudo_thresh = 0.5 if stage == 'semi' else 0.3
                        points_prev_list = generate_points_from_density(
                            teacher_pred_prev, threshold=pseudo_thresh
                        )

                    seqs_for_loss = prepare_p2r_seqs_from_points_list(points_prev_list)

                    # ── Debug: 可視化偽標籤（前 20 batch × 2 frame） ────────
                    if stage == 'semi' and batch_idx < 20:
                        if not hasattr(generate_points_from_density, '_viz_count'):
                            generate_points_from_density._viz_count = 0
                            generate_points_from_density._viz_dir = None
                        if generate_points_from_density._viz_dir is None:
                            import os as _os
                            _tag = _os.environ.get('WANDB_NAME', 'debug')
                            generate_points_from_density._viz_dir = f"debug_viz_{_tag}"
                            _os.makedirs(generate_points_from_density._viz_dir, exist_ok=True)

                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt
                        import numpy as np

                        # ── 載入該幀的 GT 點（FDST 所有幀皆有標註） ──────────
                        gt_frame = None
                        try:
                            uid_frame = uids[0]      # (video_id, frame_name)
                            _vid, _frm = uid_frame if isinstance(uid_frame, tuple) else (None, str(uid_frame))
                            _FDST_ROOT = '/media/SSD/OSshareSpace/localSpace/workPlace/zhongKe/gd/FDST'
                            _gt_path = os.path.join(
                                _FDST_ROOT, 'train_data', 'new-anno',
                                f"GT_{_vid}_{_frm}.npy" if _vid else f"GT_{_frm}.npy"
                            )
                            if os.path.exists(_gt_path):
                                gt_raw = np.load(_gt_path)[:, :2]
                                gt_frame = gt_raw
                            else:
                                print(f"[DEBUG] GT file not found: {_gt_path}", flush=True)
                        except Exception as e:
                            print(f"[DEBUG] GT load error: {e}", flush=True)

                        # 偽標籤點（augmented 空間，需 × down 映射回 256×256）
                        pseudo_pts = seqs_for_loss[0].cpu().numpy() * down if seqs_for_loss[0].numel() > 0 else []

                        # 圖片（teacher 乾淨版，augmented 256×256）
                        img_t = uimg_batch_teacher[0, t].cpu()
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img_show = (img_t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                        # 左：augmented 影像 + 偽標籤
                        ax1.imshow(img_show)
                        if len(pseudo_pts) > 0:
                            ax1.scatter(pseudo_pts[:, 0], pseudo_pts[:, 1],
                                        c='red', s=15, alpha=0.9, linewidths=0.5, edgecolors='white')
                        ax1.set_title(f"Pseudo-labels | n={len(pseudo_pts)} | thresh={pseudo_thresh}")
                        ax1.axis('off')
                        # 右：augmented 影像 + 原始 GT（若載入成功）
                        ax2.imshow(img_show)
                        if gt_frame is not None and len(gt_frame) > 0:
                            ax2.scatter(gt_frame[:, 0], gt_frame[:, 1],
                                        c='lime', s=15, alpha=0.9, linewidths=0.5, edgecolors='white',
                                        marker='o')
                        ax2.set_title(f"GT (raw .npy) | n={len(gt_frame) if gt_frame is not None else 0}")
                        ax2.axis('off')

                        plt.suptitle(f"epoch={epoch} batch={batch_idx} t={t} | "
                                     f"uid={uids[0]}", fontsize=10)
                        plt.tight_layout()
                        fname = f"{generate_points_from_density._viz_dir}/b{batch_idx}_t{t}.png"
                        plt.savefig(fname, dpi=120)
                        plt.close(fig)
                        print(f"📸 {fname} saved (pseudo={len(pseudo_pts)}, "
                              f"gt={len(gt_frame) if gt_frame is not None else 0})", flush=True)
                        generate_points_from_density._viz_count += 1
                    # ── Debug End ────────────────────────────────────────────
                    with autocast(enabled=amp_enabled):
                        loss_p2r = p2r(student_pred_t, seqs_for_loss, down=down,
                                       masks=None, pos_weight=current_pos_weight)

                    # ── Multi-task loss composition ──────────────────────────
                    # 1. Current-frame density loss (warped pseudo-labels)
                    loss_total = loss_p2r + sup_loss / max(T - 1, 1)

                    # 2. Optical-flow photometric loss (warped RGB MSE at 64×64)
                    if flo_t is not None:
                        perm_bgr = [2, 1, 0]
                        cur_rgb = denormalize(frame_t_teacher)[:, perm_bgr] / 255.0
                        prev_rgb = denormalize(
                            uimg_batch_teacher[:, (t + 1) % T],
                        )[:, perm_bgr] / 255.0
                        cur_rgb_low = F.avg_pool2d(cur_rgb, 4, 4)
                        prev_rgb_low = F.avg_pool2d(prev_rgb, 4, 4)
                        img_warp = emac_warp(prev_rgb_low, flo_t)
                        opt_loss = F.mse_loss(img_warp, cur_rgb_low) * 0.05

                        # 3. Flow smoothness (TV)
                        flow_tv = tv_loss(flo_t) * 0.01

                        loss_total = loss_total + opt_loss + flow_tv

                    batch_loss += loss_total.item()

                    if amp_enabled:
                        scaler.scale(loss_total).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss_total.backward()
                        optimizer.step()

                    update_ema(student, teacher, ema_alpha)

                    with torch.no_grad():
                        teacher_pred_fuse = teacher(
                            frame_t_teacher, templates=[uimg_batch_teacher[:, (t + 1) % T]]
                        )
                        teacher_pred_prev = teacher_pred_fuse.detach()
            else:
                check_min_batch(uimg_batch, min_bs=min_batch_size)

                student_h = student.init_hidden(B, device=device, spatial_size=(H // down, W // down))
                teacher_h = teacher.init_hidden(B, device=device, spatial_size=(H // down, W // down))

                with torch.no_grad():
                    frame0 = uimg_batch[:, 0]
                    with autocast(enabled=amp_enabled):
                        teacher_pred0, teacher_h = teacher(frame0, teacher_h)
                    teacher_pred_prev = teacher_pred0.detach()

                batch_loss = 0.0
                for t in range(1, T):
                    optimizer.zero_grad()
                    frame_t = uimg_batch[:, t]
                    flow_t_minus = None
                    if uflows_batch is not None:
                        flow_t_minus = uflows_batch[:, t - 1]

                    with autocast(enabled=amp_enabled):
                        student_pred_t, student_h = student(frame_t, student_h)

                    with torch.no_grad():
                        pseudo_thresh = 0.5 if stage == 'semi' else 0.3
                        points_prev_list = generate_points_from_density(teacher_pred_prev, threshold=pseudo_thresh)
                        if flow_t_minus is not None:
                            warped_points_list = warp_points_with_flow(points_prev_list, flow_t_minus, flow_down=2, model_down=down)
                        else:
                            warped_points_list = points_prev_list

                    seqs_for_loss = prepare_p2r_seqs_from_points_list(warped_points_list)
                    with autocast(enabled=amp_enabled):
                        loss_p2r = p2r(student_pred_t, seqs_for_loss, down=down, masks=None, pos_weight=current_pos_weight)

                    loss = loss_p2r + sup_loss / max(T - 1, 1)
                    batch_loss += loss.item()

                    if amp_enabled:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    student_h = student_h.detach()
                    update_ema(student, teacher, ema_alpha)

                    with torch.no_grad():
                        teacher_pred_t, teacher_h = teacher(frame_t, teacher_h)
                        teacher_pred_prev = teacher_pred_t.detach()

        else:
            raise ValueError(f"Unknown stage: {stage}")

        epoch_loss += batch_loss
        pbar.set_postfix(loss=f'{batch_loss:.4f}')

        del batch

    logger.info(f'Epoch [{epoch}] avg_loss: {epoch_loss / max(1, num_batches):.6f}')
    return
