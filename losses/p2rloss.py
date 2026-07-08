# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as tF


class L2DIS:
    def __init__(self, factor=512) -> None:
        self.factor = factor

    def __call__(self, X, Y):
        x_col = X.unsqueeze(-2)
        y_row = Y.unsqueeze(-3)
        C = torch.norm(x_col - y_row, dim=-1)
        C = C / self.factor
        return C


class P2RLoss(nn.modules.loss._Loss):
    def __init__(self, factor=1, reduction='mean') -> None:
        super().__init__()
        self.factor = factor
        self.cost = L2DIS(1)
        self.min_radis = 8
        self.max_radis = 96

        self.cost_class = 1
        self.cost_point = 8

    CHUNK_SIZE = 64

    def _process_chunk(self, A_chunk, B_coord_chunk, point_valid_chunk, A_coord, HW, down):
        nb = B_coord_chunk.size(0)
        max_N = B_coord_chunk.size(2)
        device = A_chunk.device

        B_flat = B_coord_chunk.view(nb, max_N, 2)
        dist2 = torch.cdist(B_flat, B_flat).pow(2)
        diag_mask = torch.eye(max_N, device=device).bool().unsqueeze(0)
        dist2 = dist2.masked_fill_(diag_mask, float('inf'))
        pad_mask = ~point_valid_chunk.view(nb, 1, max_N)
        dist2 = dist2.masked_fill_(pad_mask.expand(-1, max_N, -1), float('inf'))
        dist2 = dist2.masked_fill_(pad_mask.transpose(-1, -2).expand(-1, max_N, -1), float('inf'))
        nearest_dist = dist2.min(dim=-1, keepdim=True).values
        nearest_dist = nearest_dist.masked_fill_(point_valid_chunk.view(nb, max_N, 1).logical_not(), 32.0)

        x_col = A_coord.unsqueeze(-2)
        y_row = B_coord_chunk.unsqueeze(-3)
        C = torch.norm(x_col - y_row, dim=-1)

        with torch.no_grad():
            minC, mcidx = C.min(dim=-1, keepdim=True)
            M = torch.zeros_like(C).scatter_(-1, mcidx, 1.0) * (C < self.max_radis)
            M = M * point_valid_chunk.unsqueeze(-2)

            maxC = (minC * M).amax(dim=2, keepdim=True)
            maxC = torch.clip(maxC, min=self.min_radis, max=self.max_radis)
            C = C / maxC

            C = C * self.cost_point - A_chunk.view(nb, 1, HW, 1) * self.cost_class

            vid = (M.sum(dim=2) > 0) & point_valid_chunk
            C = C * vid.unsqueeze(-2).float()

            C2 = M * C + (1 - M) * (C.max() + 1)
            minC2, mcidx2 = C2.min(dim=2, keepdim=True)
            T_chunk = torch.zeros_like(C2).scatter_(2, mcidx2, 1.0).sum(dim=-1)
            T_chunk = (T_chunk > 0.5).float()
            W_chunk = T_chunk + 1.0

        return T_chunk.view(nb, HW), W_chunk.view(nb, HW)

    def forward(self, dens, seqs, down, masks=None, crop_den_masks=None):
        bs = len(seqs)
        if bs == 0:
            return torch.tensor(0.0, device=dens.device)

        H, W = dens.shape[2], dens.shape[3]
        device = dens.device
        HW = H * W

        A_coord = torch.stack(torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'), dim=-1).float().view(1, 1, HW, 2) * down + (down - 1) / 2
        A = dens.view(bs, HW)

        T_full = torch.zeros(bs, HW, device=device)
        W_full = torch.ones(bs, HW, device=device) * 0.5

        nonempty_idx = [i for i in range(bs) if seqs[i].size(0) >= 1]
        if nonempty_idx:
            nb = len(nonempty_idx)
            seqs_list = [seqs[i] for i in nonempty_idx]
            max_N = max(s.size(0) for s in seqs_list)

            # Process in chunks to limit GPU memory (C, M, C2 scale as nb*HW*max_N)
            for chunk_start in range(0, nb, self.CHUNK_SIZE):
                chunk_end = min(chunk_start + self.CHUNK_SIZE, nb)
                chunk_idx = nonempty_idx[chunk_start:chunk_end]
                chunk_seqs = seqs_list[chunk_start:chunk_end]
                cnb = chunk_end - chunk_start
                c_max_N = max(s.size(0) for s in chunk_seqs)

                B_coord_chunk = torch.zeros(cnb, 1, c_max_N, 2, device=device)
                point_valid_chunk = torch.zeros(cnb, 1, c_max_N, dtype=torch.bool, device=device)
                for j, s in enumerate(chunk_seqs):
                    n = s.size(0)
                    B_coord_chunk[j, 0, :n] = s[:, :2].float()
                    point_valid_chunk[j, 0, :n] = True

                T_chunk, W_chunk = self._process_chunk(
                    A[chunk_idx], B_coord_chunk, point_valid_chunk,
                    A_coord, HW, down)

                T_full[chunk_idx] = T_chunk
                W_full[chunk_idx] = W_chunk

        loss = tF.binary_cross_entropy_with_logits(A, T_full, weight=W_full, reduction='mean')
        return loss
    
