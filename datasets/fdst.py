# -*- coding: utf-8 -*-

"""
datasets/fdst.py

Flexible FDST (Fudan-ShanghaiTech / video) dataset adapter that returns sequences of frames
and optional precomputed optical flow per-pair.

Designed to be tolerant to a few common directory layouts:
- images stored as per-video subfolders:
    <root>/<mode>_data/images/<video_id>/<frame_name>.jpg
- or flat layout:
    <root>/<mode>_data/images/<frame_name>.jpg

Annotations: the code will attempt several filename patterns to locate per-frame point annotations (.npy):
- <root>/<mode>_data/new-anno/GT_{video}_{frame}.npy
- <root>/<mode>_data/new-anno/GT_{frame}.npy
- <root>/<mode>_data/annotations/{video}/{frame}.npy
- <root>/<mode>_data/annotations/{frame}.npy

Flow files: expected under flow_root with patterns like:
- {video}_{frame}_flow.npy or {frame}_flow.npy

If you need a different layout, pass flow_root and keep filenames consistent with one of the patterns above.

This implementation mirrors the sequence/flow API introduced for SHHA: returns sequences of frames shaped
(T, C, H, W) and collate packs batches into (B, T, C, H, W). The collate_fn is compatible with train loop.
"""

import os
import numpy as np
import torch
from torch.utils import data
from PIL import Image
import random

if __name__ == '__main__':
    from utils import NormalSample
else:
    from .utils import NormalSample


class FDST(data.Dataset):
    def __init__(self, root_path, mode, label_prob=1, protc_path='', seq_len=1, seq_stride=1, flow_root=None, flow_ext='.npy'):
        self.training = (mode == 'train')
        self.root_path = root_path
        self.mode = mode
        self.seq_len = int(seq_len)
        self.seq_stride = int(seq_stride)
        self.flow_root = flow_root
        self.flow_ext = flow_ext

        images_dir = os.path.join(root_path, mode + '_data', 'images')
        assert os.path.exists(images_dir), f"images dir not found: {images_dir}"

        # detect whether images are organized per-video (subdirs) or flat
        entries = [e for e in os.listdir(images_dir) if not e.startswith('.')]
        has_subdirs = any(os.path.isdir(os.path.join(images_dir, e)) for e in entries)

        self.frames = []  # list of tuples (video_id_or_none, frame_basename)
        if has_subdirs:
            # iterate videos
            vids = sorted([d for d in entries if os.path.isdir(os.path.join(images_dir, d))])
            for vid in vids:
                vid_dir = os.path.join(images_dir, vid)
                frames = sorted([f for f in os.listdir(vid_dir) if f.lower().endswith(('jpg','png','jpeg'))])
                for f in frames:
                    name = os.path.splitext(f)[0]
                    self.frames.append((vid, name))
        else:
            frames = sorted([f for f in entries if f.lower().endswith(('jpg','png','jpeg'))])
            for f in frames:
                name = os.path.splitext(f)[0]
                self.frames.append((None, name))

        # build index lists for labeled/unlabeled frames
        self.label = []
        self.unlabel = []

        # protocol file may list ids as either 'video/frame' or 'frame'
        idset = None
        if protc_path:
            with open(protc_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            idset = set(lines)

        # populate label/unlabel based on protocol if provided, otherwise treat all as labeled (if not training) or unlabeled
        for vid, frm in self.frames:
            key1 = f"{vid}/{frm}" if vid is not None else frm
            key2 = frm
            if (not self.training) or (idset is None) or (key1 in idset) or (key2 in idset):
                self.label.append((vid, frm))
            self.unlabel.append((vid, frm))

        # O(1) lookup dictionaries replacing O(N) .index() calls
        self.label_to_idx = {tpl: i for i, tpl in enumerate(self.label)}
        self.unlabel_to_idx = {tpl: i for i, tpl in enumerate(self.unlabel)}

        self.img_dir = images_dir
        # annotation directories to try
        self.dot_dirs = (
            os.path.join(root_path, mode + '_data', 'new-anno'),
            os.path.join(root_path, mode + '_data', 'annotations'),
        )

        self.norm_func = NormalSample(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            crop_size=(256, 256),
            train=self.training
        )

        print(f"[FDST training={self.training}]: {len(self.label)} labeled frames, {len(self.unlabel)} total frames")

    def __len__(self):
        return len(self.unlabel) if self.training else len(self.label)

    def __getitem__(self, index):
        if self.training:
            # labeled sequence: sample a labeled frame index
            lid = random.choice(self.label)
            limg_seq, lseqs = self.readLabelSequenceFromTuple(lid)
            uid = self.unlabel[index % len(self.unlabel)]
            uimg_seq, umask_seq = self.readUnlabelSequenceFromTuple(uid)
            uflows = self._load_flow_for_sequence(self.unlabel, index % len(self.unlabel))
            return limg_seq, lseqs, lid, uimg_seq, umask_seq, uid, uflows
        else:
            lid = self.label[index]
            limg_seq, lseqs = self.readLabelSequenceFromTuple(lid)
            flows = self._load_flow_for_sequence(self.label, index)
            return limg_seq, lseqs, lid, flows

    def _get_sequence_indices(self, list_ref, start_idx):
        seq_idx = []
        start = start_idx
        for i in range(self.seq_len):
            idx = start + i * self.seq_stride
            if idx >= len(list_ref):
                idx = len(list_ref) - 1
            seq_idx.append(idx)
        return seq_idx

    def readLabelSequenceFromTuple(self, tpl):
        # tpl: (vid_or_none, frame_name)
        if tpl not in self.label_to_idx:
            # fallback: single frame read
            return self._read_single_label_tpl(tpl)
        idx = self.label_to_idx[tpl]
        idxs = self._get_sequence_indices(self.label, idx)
        imgs = []
        dotseqs = []
        for ii in idxs:
            vid, frm = self.label[ii]
            img = self._load_image(vid, frm)
            img_t = self.norm_func.im2tensor(img)
            dot = self._load_annotation(vid, frm)
            img_t, dot = self.norm_func.process_lable(img_t, dot)
            imgs.append(img_t.squeeze(0))
            dotseqs.append(dot)
        imgs_seq = torch.stack(imgs, dim=0)
        return imgs_seq, dotseqs

    def readUnlabelSequenceFromTuple(self, tpl):
        if tpl not in self.unlabel_to_idx:
            return self._read_single_unlabel_tpl(tpl)
        idx = self.unlabel_to_idx[tpl]
        idxs = self._get_sequence_indices(self.unlabel, idx)
        imgs = []
        masks = []
        for ii in idxs:
            vid, frm = self.unlabel[ii]
            img = self._load_image(vid, frm)
            wa_img = self.norm_func.im2tensor(img)
            img_proc = self.norm_func.process_unlabel(wa_img)
            imgs.append(img_proc.squeeze(0))
            masks.append(self.random_mask(img_proc).squeeze(0))
        imgs_seq = torch.stack(imgs, dim=0)
        masks_seq = torch.stack(masks, dim=0)
        return imgs_seq, masks_seq

    def _load_image(self, vid, frm):
        if vid is None:
            imgpath = os.path.join(self.img_dir, frm + '.jpg')
        else:
            imgpath = os.path.join(self.img_dir, vid, frm + '.jpg')
        if not os.path.exists(imgpath):
            if vid is None:
                imgpath = os.path.join(self.img_dir, frm + '.png')
            else:
                imgpath = os.path.join(self.img_dir, vid, frm + '.png')
            if not os.path.exists(imgpath):
                raise FileNotFoundError(f"Image not found: {imgpath}")
        return Image.open(imgpath).convert('RGB')

    def _load_annotation(self, vid, frm):
        candidates = []
        for d in self.dot_dirs:
            candidates.append(os.path.join(d, f"GT_{vid}_{frm}.npy") if vid is not None else os.path.join(d, f"GT_{frm}.npy"))
            candidates.append(os.path.join(d, vid, f"{frm}.npy") if vid is not None else os.path.join(d, f"{frm}.npy"))
        for p in candidates:
            if os.path.exists(p):
                return torch.from_numpy(np.load(p, mmap_mode='r'))[:, :2]
        return torch.zeros((0, 2), dtype=torch.float32)

    def _read_single_label_tpl(self, tpl):
        vid, frm = tpl
        img = self._load_image(vid, frm)
        img_t = self.norm_func.im2tensor(img)
        dot = self._load_annotation(vid, frm)
        img_t, dot = self.norm_func.process_lable(img_t, dot)
        return img_t, [dot]

    def _read_single_unlabel_tpl(self, tpl):
        vid, frm = tpl
        img = self._load_image(vid, frm)
        wa_img = self.norm_func.im2tensor(img)
        img_proc = self.norm_func.process_unlabel(wa_img)
        mask = self.random_mask(img_proc).squeeze(0)
        return img_proc, mask

    def _load_flow_for_sequence(self, list_ref, start_idx):
        if not self.flow_root:
            return None
        idxs = self._get_sequence_indices(list_ref, start_idx)
        flows = []
        for i in range(len(idxs) - 1):
            vid, frm = list_ref[idxs[i]]
            # try multiple filename conventions
            cand1 = os.path.join(self.flow_root, f"{vid}_{frm}_flow{self.flow_ext}") if vid is not None else os.path.join(self.flow_root, f"{frm}_flow{self.flow_ext}")
            cand2 = os.path.join(self.flow_root, f"{frm}_flow{self.flow_ext}")
            fn = None
            for c in (cand1, cand2):
                if c and os.path.exists(c):
                    fn = c
                    break
            if fn is None:
                flows.append(None)
                continue
            if self.flow_ext == '.npy':
                arr = np.load(fn, mmap_mode='r')
                flows.append(torch.from_numpy(arr).float())
            else:
                obj = torch.load(fn, map_location='cpu')
                flows.append(obj if isinstance(obj, torch.Tensor) else torch.from_numpy(np.asarray(obj)).float())
        if len(flows) == 0:
            return None
        first = next((f for f in flows if f is not None), None)
        if first is None:
            return None
        stacked = []
        for f in flows:
            if f is None:
                stacked.append(torch.zeros_like(first))
            else:
                stacked.append(f)
        return torch.stack(stacked, dim=0)

    def random_mask(self, uimgs):
        bsize, _, img_h, img_w = uimgs.shape
        cut_img_mask = torch.ones((bsize, 1, img_h, img_w))
        min_cut, max_cut = 1 / 8, 1 / 4
        for i in range(bsize):
            cut_w = int(img_w * (min_cut + random.random() * (max_cut - min_cut)))
            cut_h = int(img_h * (min_cut + random.random() * (max_cut - min_cut)))
            cut_top = random.randint(0, img_h - cut_h)
            cut_left = random.randint(0, img_w - cut_w)
            cut_bottom, cut_right = cut_top + cut_h, cut_left + cut_w
            cut_img_mask[i, :, cut_top:cut_bottom, cut_left:cut_right] = 0
        return cut_img_mask

    @staticmethod
    def collate_fn(samples):
        # training case returns: limg_seq, lseqs, lids, uimg_seq, umask_seq, uids, uflows
        if len(samples[0]) >= 7:
            limg_seqs, lseqs, lids, uimg_seqs, umask_seqs, uids, uflows = zip(*samples)
            limg_batch = torch.stack(limg_seqs, dim=0)
            uimg_batch = torch.stack(uimg_seqs, dim=0)
            umask_batch = torch.stack(umask_seqs, dim=0)
            if all([f is None for f in uflows]):
                uflows_batch = None
            else:
                first = next(f for f in uflows if f is not None)
                stacked_flows = []
                for f in uflows:
                    if f is None:
                        stacked_flows.append(torch.zeros_like(first))
                    else:
                        stacked_flows.append(f)
                uflows_batch = torch.stack(stacked_flows, dim=0)
            return limg_batch, lseqs, lids, uimg_batch, umask_batch, uids, uflows_batch
        else:
            # eval case: limg_seq, lseqs, lid, flows(optional)
            if len(samples[0]) == 3:
                limg_seqs, lseqs, lids = zip(*samples)
                limg_batch = torch.stack(limg_seqs, dim=0)
                return limg_batch, lseqs, lids
            else:
                limg_seqs, lseqs, lids, flows = zip(*samples)
                limg_batch = torch.stack(limg_seqs, dim=0)
                if all([f is None for f in flows]):
                    flows_batch = None
                else:
                    first = next(f for f in flows if f is not None)
                    stacked = []
                    for f in flows:
                        if f is None:
                            stacked.append(torch.zeros_like(first))
                        else:
                            stacked.append(f)
                    flows_batch = torch.stack(stacked, dim=0)
                return limg_batch, lseqs, lids, flows_batch


if __name__ == '__main__':
    # minimal local test
    datadir = "/data/FDST"
    data = FDST(datadir, 'train', seq_len=3)
    from torch.utils.data import DataLoader
    loader = DataLoader(data, batch_size=2, num_workers=1, collate_fn=FDST.collate_fn)
    for b in loader:
        print([type(x) for x in b])
        break
