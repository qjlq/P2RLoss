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


def sample_flow_at_points(flow, points):
    """
    Sample per-point flow vectors from a dense flow map using grid_sample.
    Args:
        flow: Tensor (B, 2, H, W)
        points: Tensor (B, N, 2) in (x, y) pixel coordinates (0..W-1, 0..H-1). If N==0 returns empty tensor.
    Returns:
        sampled: Tensor (B, N, 2) flow vectors (dx, dy) in pixels
    """
    B, _, H, W = flow.shape
    device = flow.device
    sampled_list = []
    for b in range(B):
        pts = points[b]  # (N,2)
        if pts.numel() == 0:
            sampled_list.append(torch.zeros((0, 2), device=device))
            continue
        xs = pts[:, 0]
        ys = pts[:, 1]
        # normalize to [-1,1]
        nx = (xs / float(max(W - 1, 1))) * 2.0 - 1.0
        ny = (ys / float(max(H - 1, 1))) * 2.0 - 1.0
        grid = torch.stack((nx, ny), dim=1).unsqueeze(0).unsqueeze(0)  # (1,1,N,2)
        # grid_sample expects (N,H_out,W_out,2) style, so we reshape flow to (1,2,H,W)
        f = flow[b:b+1]
        sampled = F.grid_sample(f, grid.view(1, -1, 1, 2), align_corners=True).view(2, -1).permute(1, 0)  # (N,2)
        sampled_list.append(sampled)
    # pad to same N across batch by returning list of tensors
    # For convenience, return list of (N_i,2) tensors per batch
    return sampled_list


def warp_points_with_flow(points_list, flow_batch):
    """
    Warp a list of points (per-batch) using flow maps for corresponding frame-pairs.
    Args:
        points_list: list length B of tensors (N_i,2) in (x,y) pixels for previous frame
        flow_batch: Tensor (B, 2, H, W) flow from prev->curr (dx,dy) in pixels
    Returns:
        warped_list: list length B of tensors (N_i,2) warped to current frame coordinates
    """
    B = flow_batch.shape[0]
    device = flow_batch.device
    # prepare points tensor padded for sampling
    maxN = max([p.shape[0] for p in points_list]) if len(points_list) > 0 else 0
    if maxN == 0:
        return [torch.zeros((0, 2), device=device) for _ in range(B)]
    # build (B, N, 2) with padding (duplicate last or zeros)
    pts_padded = torch.zeros((B, maxN, 2), device=device)
    mask = torch.zeros((B, maxN), dtype=torch.bool, device=device)
    for b in range(B):
        n = points_list[b].shape[0]
        if n > 0:
            pts_padded[b, :n] = points_list[b].to(device)
            mask[b, :n] = True
    sampled_list = sample_flow_at_points(flow_batch, pts_padded)  # returns list per batch
    warped = []
    for b in range(B):
        if points_list[b].shape[0] == 0:
            warped.append(points_list[b].new_zeros((0, 2)))
            continue
        flow_vecs = sampled_list[b].to(points_list[b].device)
        warped_pts = points_list[b] + flow_vecs
        warped.append(warped_pts)
    return warped


def prepare_p2r_seqs_from_points_list(points_list):
    """
    Convert list of per-batch (N_i,2) into list of sequences expected by P2RLoss
    P2RLoss expects seqs as a list where each element corresponds to sample i and is a Tensor (Ni,2)
    So we simply return the points_list as-is (ensuring float and device consistency)
    """
    return [p.float() for p in points_list]


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

    logger = logging.getLogger('VGG16BN')
    num_batches = len(dataloader)
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

        # === Supervised loss on labeled first frame ===
        sup_loss = 0.0
        if limg_batch is not None and limg_batch.size(0) > 0:
            lframe = limg_batch[:, 0]
            with autocast(enabled=amp_enabled):
                lpred, _ = student(lframe, prev_h=None)
            ldots = [seq[0][0][:, :2].to(device) for seq in lseqs]
            sup_loss = p2r(lpred, ldots, down=down)

        if stage == 'sup':
            # Supervised only: backward on supervised loss
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
            # Semi-supervised: supervised + temporal pseudo-labeling
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
                    points_prev_list = generate_points_from_density(teacher_pred_prev, threshold=0.3)
                    if flow_t_minus is not None:
                        warped_points_list = warp_points_with_flow(points_prev_list, flow_t_minus)
                    else:
                        warped_points_list = points_prev_list

                seqs_for_loss = prepare_p2r_seqs_from_points_list(warped_points_list)
                with autocast(enabled=amp_enabled):
                    loss_p2r = p2r(student_pred_t, seqs_for_loss, down=down, masks=None)

                loss = loss_p2r + sup_loss / (T - 1)
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

    logger.info(f'Epoch [{epoch}] avg_loss: {epoch_loss / max(1, num_batches):.6f}')
    return
