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
    Args:
        density_logits: Tensor (B,1,H,W) logits (before sigmoid)
    Returns:
        list_of_points: length B list, each is Tensor (N_i, 2) in (x, y) image coordinates (pixel indices)
    """
    prob = torch.sigmoid(density_logits)
    B, C, H, W = prob.shape
    assert C == 1
    pool = F.max_pool2d(prob, kernel_size=3, stride=1, padding=1)
    peaks = (prob == pool) & (prob > threshold)
    points_list = []
    for b in range(B):
        ys, xs = torch.nonzero(peaks[b, 0], as_tuple=True)
        if xs.numel() == 0:
            points_list.append(torch.zeros((0, 2), device=density_logits.device, dtype=torch.float32))
            continue
        coords = torch.stack([xs.float(), ys.float()], dim=1)  # (N, 2) as (x, y)
        if topk is not None and coords.shape[0] > topk:
            # choose topk by probability
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
                        pseudo_thresh = 0.85 if stage == 'semi' else 0.3
                        points_prev_list = generate_points_from_density(
                            teacher_pred_prev, threshold=pseudo_thresh
                        )

                    seqs_for_loss = prepare_p2r_seqs_from_points_list(points_prev_list)
                    with autocast(enabled=amp_enabled):
                        loss_p2r = p2r(student_pred_t, seqs_for_loss, down=down,
                                       masks=None, pos_weight=current_pos_weight)

                    # ── Multi-task loss composition ──────────────────────────
                    # 1. Current-frame density loss (warped pseudo-labels)
                    loss_total = loss_p2r + sup_loss / max(T - 1, 1)

                    # 2. Optical-flow photometric loss (warped RGB MSE)
                    if flo_t is not None:
                        perm_bgr = [2, 1, 0]
                        cur_rgb = denormalize(frame_t_teacher)[:, perm_bgr] / 255.0
                        prev_rgb = denormalize(
                            uimg_batch_teacher[:, (t + 1) % T],
                        )[:, perm_bgr] / 255.0
                        img_warp = emac_warp(prev_rgb, flo_t)
                        opt_loss = F.mse_loss(img_warp, cur_rgb) * 0.05

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
                        pseudo_thresh = 0.85 if stage == 'semi' else 0.3
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
