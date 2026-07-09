# -*- coding: utf-8 -*-

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_descriptor')

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
if rlimit[0] < 65536:
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (65536, rlimit[1]))
    except (ValueError, resource.error):
        pass

from torch.utils.data import DataLoader
from .shha import SHHA
from .fdst import FDST

def build_loader(config, mode):
    data_path = config.DATA_PATH
    label_prob = getattr(config, 'LABEL_PERCENT', 1.0)
    protc_path = getattr(config, 'LABEL_PROTOCOL', '')
    batch_size = config.BATCH_SIZE
    num_workers = config.NUM_WORKERS

    Dataset = {
        'shha': SHHA,
        'fdst': FDST,
    }[config.DATASET.lower()]

    # sequence and flow options
    seq_len = getattr(config.DATA, 'SEQ_LEN', 1) if hasattr(config, 'DATA') else getattr(config, 'SEQ_LEN', 1)
    seq_stride = getattr(config.DATA, 'SEQ_STRIDE', 1) if hasattr(config, 'DATA') else getattr(config, 'SEQ_STRIDE', 1)
    flow_root = getattr(config.DATA, 'FLOW_ROOT', None) if hasattr(config, 'DATA') else None
    flow_ext = getattr(config.DATA, 'FLOW_EXT', '.npy') if hasattr(config, 'DATA') else '.npy'

    data_set = Dataset(data_path, mode, label_prob, protc_path, seq_len=seq_len, seq_stride=seq_stride, flow_root=flow_root, flow_ext=flow_ext)

    return DataLoader(
        data_set,
        batch_size = batch_size if (mode == 'train') else 1,
        num_workers = num_workers,
        pin_memory = True if (mode == 'train') else False,
        shuffle = (mode == 'train'),
        collate_fn=Dataset.collate_fn,
        persistent_workers = num_workers > 0,
        prefetch_factor = 4 if num_workers > 0 else 2,
    )

def build_normal_loader(config, mode):
    data_path = config.DATA_PATH
    batch_size = config.BATCH_SIZE
    
    Dataset = {
        'shha': SHHA,
        'fdst': FDST
    }[config.DATASET.lower()]
    
    data_set = Dataset(data_path, mode)

    return DataLoader(
        data_set,
        batch_size = batch_size,
        num_workers = 4,
        pin_memory=config.PIN_MEMORY,
        shuffle = False,
        collate_fn=Dataset.collate_fn
    )
