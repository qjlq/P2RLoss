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

            B_coord_batch = torch.zeros(nb, 1, max_N, 2, device=device)
            point_valid = torch.zeros(nb, 1, max_N, dtype=torch.bool, device=device)
            for j, s in enumerate(seqs_list):
                n = s.size(0)
                B_coord_batch[j, 0, :n] = s[:, :2].float()
                point_valid[j, 0, :n] = True

            x_col = A_coord.unsqueeze(-2)
            y_row = B_coord_batch.unsqueeze(-3)
            C = torch.norm(x_col - y_row, dim=-1)

            with torch.no_grad():
                minC, mcidx = C.min(dim=-1, keepdim=True)
                M = torch.zeros_like(C).scatter_(-1, mcidx, 1.0) * (C < self.max_radis)
                M = M * point_valid.unsqueeze(-2)

                maxC = (minC * M).amax(dim=2, keepdim=True)
                maxC = torch.clip(maxC, min=self.min_radis, max=self.max_radis)
                C = C / maxC

                C = C * self.cost_point - (A[nonempty_idx].view(nb, 1, HW, 1) * self.cost_class)

                vid = (M.sum(dim=2) > 0) & point_valid
                C = C * vid.unsqueeze(-2).float()

                C2 = M * C + (1 - M) * (C.max() + 1)
                minC2, mcidx2 = C2.min(dim=2, keepdim=True)
                T_nb = torch.zeros_like(C2).scatter_(2, mcidx2, 1.0).sum(dim=-1)
                T_nb = (T_nb > 0.5).float()
                W_nb = T_nb + 1.0

                T_full[nonempty_idx] = T_nb.view(nb, HW)
                W_full[nonempty_idx] = W_nb.view(nb, HW)

        loss = tF.binary_cross_entropy_with_logits(A, T_full, weight=W_full, reduction='mean')
        return loss
    
