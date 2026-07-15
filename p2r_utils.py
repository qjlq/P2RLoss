# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random

def load_checkpoint(config, model, optimizer, lr_scheduler, scaler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    teacher, student = model
    msg_t = teacher.load_state_dict(checkpoint['teacher'], strict=False)
    logger.info(f"[load teacher]: {msg_t}")
    msg_s = student.load_state_dict(checkpoint['student'], strict=False)
    logger.info(f"[load student]: {msg_s}")
    if optimizer is not None and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("[load optimizer]: OK")
        except (ValueError, KeyError) as e:
            logger.warning(f"[load optimizer]: SKIPPED (group mismatch: {e})")
    if lr_scheduler is not None and 'lr_scheduler' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        logger.info("[load lr_scheduler]: OK")
    if scaler is not None and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
        logger.info("[load scaler]: OK")
    saved_epoch = checkpoint.get('epoch', -1)
    if isinstance(saved_epoch, str):
        import re
        match = re.search(r'(\d+)$', saved_epoch)
        saved_epoch = int(match.group(1)) if match else -1
    max_accuracy = checkpoint.get('max_accuracy', [1e6] * 3)
    return saved_epoch, max_accuracy

def save_checkpoint(config, epoch, model, optimizer, lr_scheduler, scaler, max_accuracy, logger):
    teacher, student = model
    # Extract numeric epoch for checkpoint metadata (handles "best_epoch25" → 25)
    if isinstance(epoch, str):
        import re
        match = re.search(r'(\d+)$', epoch)
        numeric_epoch = int(match.group(1)) if match else -1
    else:
        numeric_epoch = epoch
    save_state = {
        'teacher': teacher.state_dict(),
        'student': student.state_dict(),
        'epoch': numeric_epoch,
        'optimizer': optimizer.state_dict() if optimizer else None,
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
        'scaler': scaler.state_dict() if scaler else None,
        'max_accuracy': max_accuracy,
    }
    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def plot_curve(label, epo, data, savepath):
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig = plt.figure()
    plt.title(label)
    plt.plot(epo, data)
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.grid(True)
    plt.savefig(savepath)
    plt.close(fig)

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_optimizer_groups(model, base_lr, backbone_lr, backbone_keywords=None):
    """Generic optimizer param group factory.

    Automatically splits model parameters into two groups based on
    ``backbone_keywords``.  Parameters whose name contains any of the
    keywords get ``backbone_lr``; all others get ``base_lr``.
    Parameters with ``requires_grad=False`` are excluded.

    Args:
        model:         nn.Module instance
        base_lr:       learning rate for non-backbone (head) parameters
        backbone_lr:   learning rate for backbone parameters
        backbone_keywords: list of substrings to identify backbone params.
                          If None or empty, all params are in a single group.

    Returns:
        list of dicts suitable for ``optim.Adam(param_groups, ...)``
    """
    if not backbone_keywords:
        trainable = [p for p in model.parameters() if p.requires_grad]
        return [{"params": trainable}]

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(kw in name for kw in backbone_keywords):
            backbone_params.append(param)
        else:
            head_params.append(param)

    groups = []
    if head_params:
        groups.append({"params": head_params})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})

    if not groups:
        raise RuntimeError(
            "create_optimizer_groups: no trainable parameters found "
            f"(backbone_keywords={backbone_keywords})"
        )
    return groups