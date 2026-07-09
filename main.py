@torch.no_grad()
def validate(config, data_loader, model, criterion, peak_thresh=0.50, nms_kernel=3):
    """
    Temporal validation:
      - keep full sequence (B,T,C,H,W)
      - propagate hidden state prev_h across time
      - count via local maxima on sigmoid(logits), not by (logit > 0) pixel area
    """
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()
    mse_meter = AverageMeter()

    end = time.time()

    for idx, batch in enumerate(data_loader):
        # Support old/new loader format
        if len(batch) == 3:
            images, dotseq, imgid = batch
        else:
            images, dotseq, imgid = batch[0], batch[1], batch[2]

        images = images.cuda(non_blocking=True)

        # Build GT count from first supervised frame labels (same convention as your current code)
        # dotseq entry expected per-sample; robustly handle nested list/tuple
        gt_counts = []
        for s in dotseq:
            if isinstance(s, (list, tuple)):
                # e.g. s[0][0] in your original; keep robust fallback
                if len(s) > 0 and isinstance(s[0], (list, tuple)):
                    cur = s[0][0]
                elif len(s) > 0:
                    cur = s[0]
                else:
                    cur = torch.empty((0, 2))
            else:
                cur = s
            gt_counts.append(cur.cuda(non_blocking=True))
        cnt = torch.tensor([d.size(0) for d in gt_counts], dtype=torch.float32, device=images.device)

        bsize = images.size(0)

        # Ensure temporal shape
        if images.dim() == 4:
            images = images.unsqueeze(1)  # (B,1,C,H,W)

        B, T, C, H, W = images.shape
        prev_h = None
        last_logits = None
        seq_loss = 0.0

        # Temporal forward across all frames
        for t in range(T):
            frame = images[:, t]  # (B,C,H,W)
            if hasattr(model, 'init_hidden'):
                logits_t, prev_h = model(frame, prev_h=prev_h)
            else:
                logits_t = model(frame)
            last_logits = logits_t

            # Optional: average frame losses against same GT supervision proxy
            # (keeps validation loss numerically stable for temporal models)
            seq_loss += criterion(logits_t, gt_counts, frame.size(-1) // logits_t.size(-1)).item()

        loss = seq_loss / max(T, 1)

        # ---- Peak-based counting (NMS via max-pool on probability map) ----
        prob = torch.sigmoid(last_logits)  # (B,1,h,w) or (B,C,h,w)
        if prob.dim() == 3:
            prob = prob.unsqueeze(1)

        pooled = tF.max_pool2d(
            prob,
            kernel_size=nms_kernel,
            stride=1,
            padding=nms_kernel // 2
        )
        peak_mask = (prob == pooled) & (prob >= peak_thresh)
        outnum = peak_mask.sum(dim=(1, 2, 3)).float()

        diff = torch.abs(outnum - cnt)
        mae = diff.mean()
        mse = (diff ** 2).mean()

        loss_meter.update(loss, bsize)
        mae_meter.update(mae.item(), bsize)
        mse_meter.update(mse.item(), bsize)

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]  '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                f'Loss {loss_meter.val:.6f} ({loss_meter.avg:.6f})  '
                f'MAE {mae_meter.val:.3f} ({mae_meter.avg:.3f})  '
                f'MSE {mse_meter.val ** 0.5:.3f} ({mse_meter.avg ** 0.5:.3f})  '
                f'Mem {memory_used:.0f}MB')

    logger.info(f' * MAE {mae_meter.avg:.3f} MSE {mse_meter.avg ** 0.5:.3f}')
    return mae_meter.avg, mse_meter.avg ** 0.5, loss_meter.avg