# -*- coding: utf-8 -*-

import numpy as np
import os
import torch
import torch.nn.functional as tF
from torch.utils import data
from PIL import Image
import tqdm
import random

if __name__ == '__main__':
    from utils import NormalSample
else:
    from .utils import NormalSample


class SHHA(data.Dataset):
    def __init__(self, root_path, mode, label_prob=1, protc_path='', seq_len=1, seq_stride=1, flow_root=None, flow_ext='.npy'):
        self.training = (mode == 'train')
        self.label, self.unlabel = [], []
        
        assert protc_path != '', f"protocol path is invalid: {protc_path}"
        with open(protc_path) as f:
            imgids = f.read().strip().split()
            imgids = set(imgids) # [:int(len(imgids) * label_prob)]
        

        imtype = 'jpg'
        for imgf in os.listdir(os.path.join(root_path, mode + '_data', 'images')):
            if not imgf.endswith(imtype):
                continue
            if (not self.training) or (imgf in imgids):
                    self.label.append(imgf.replace('.' + imtype, ''))
            self.unlabel.append(imgf.replace('.' + imtype, ''))
        
        # sort to ensure temporal ordering
        self.unlabel = sorted(self.unlabel)
        self.label = sorted(self.label)
        
        self.imgpath = os.path.join(root_path, mode + '_data', 'images', '{}' + f'.{imtype}')
        self.dotpath = os.path.join(root_path, mode + '_data', 'new-anno', 'GT_{}.npy')
        
        self.norm_func = NormalSample(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            crop_size=(256, 256),
            train = self.training
        )

        self.seq_len = int(seq_len)
        self.seq_stride = int(seq_stride)
        self.flow_root = flow_root
        self.flow_ext = flow_ext

        print(f"[training = {self.training}]: {len(self.label)} imgs are labeled &  {len(self.unlabel)} imgs are unlabeled.")

    def __len__(self):
        return  len(self.unlabel) if self.training else len(self.label)

    def __getitem__(self, index):
        if self.training:
            lid = random.choice(self.label)
            limg_seq, lseqs = self.readLabelSequenceFromId(lid)
            uid = self.unlabel[index % len(self.unlabel)]
            uimg_seq, umask_seq = self.readUnlabelSequenceFromId(uid)
            uflows = self._load_flow_for_sequence(self.unlabel, index % len(self.unlabel))
            return limg_seq, lseqs, lid, uimg_seq, umask_seq, uid, uflows
        else:
            lid = self.label[index]
            limg_seq, lseqs = self.readLabelSequenceFromId(lid)
            flows = self._load_flow_for_sequence(self.label, index)
            return limg_seq, lseqs, lid, flows

    def _get_sequence_indices_from_list(self, ordered_list, start_idx):
        seq_idx = []
        start = start_idx
        for i in range(self.seq_len):
            idx = start + i * self.seq_stride
            if idx >= len(ordered_list):
                idx = len(ordered_list) - 1
            seq_idx.append(idx)
        return seq_idx

    def readLabelSequenceFromId(self, smpid):
        # smpid is a frame id (string). We will locate its index in self.label and build a forward sequence
        if smpid not in self.label:
            # fallback: single-frame read
            return self.read_single_label(smpid)
        idx = self.label.index(smpid)
        idxs = self._get_sequence_indices_from_list(self.label, idx)
        imgs = []
        dotseqs = []
        for ii in idxs:
            sid = self.label[ii]
            imgpath = self.imgpath.format(sid)
            img = Image.open(imgpath).convert('RGB')
            img_t = self.norm_func.im2tensor(img)
            dot = torch.from_numpy(np.load(self.dotpath.format(sid)))[:, :2]
            img_t, dot = self.norm_func.process_lable(img_t, dot)
            imgs.append(img_t.squeeze(0))
            dotseqs.append(dot)
        imgs_seq = torch.stack(imgs, dim=0)  # (T, C, H, W)
        return imgs_seq, dotseqs

    def readUnlabelSequenceFromId(self, smpid):
        if smpid not in self.unlabel:
            return self.read_single_unlabel(smpid)
        idx = self.unlabel.index(smpid)
        idxs = self._get_sequence_indices_from_list(self.unlabel, idx)
        imgs = []
        masks = []
        for ii in idxs:
            sid = self.unlabel[ii]
            imgpath = self.imgpath.format(sid)
            img_raw = Image.open(imgpath).convert('RGB')
            wa_img = self.norm_func.im2tensor(img_raw)
            sa_img = self.norm_func.strong_aug(img_raw)
            img_proc = self.norm_func.process_unlabel(wa_img)
            imgs.append(img_proc.squeeze(0))
            masks.append(self.random_mask(img_proc).squeeze(0))
        imgs_seq = torch.stack(imgs, dim=0)
        masks_seq = torch.stack(masks, dim=0)
        return imgs_seq, masks_seq

    def read_single_label(self, smpid):
        imgpath = self.imgpath.format(smpid)
        img = Image.open(imgpath).convert('RGB')
        img = self.norm_func.im2tensor(img)
        dotseq = torch.from_numpy(np.load(self.dotpath.format(smpid)))[:, :2]
        img, dotseq = self.norm_func.process_lable(img, dotseq)
        return img, [dotseq]

    def read_single_unlabel(self, smpid):
        imgpath = self.imgpath.format(smpid)
        img_raw = Image.open(imgpath).convert('RGB')
        wa_img = self.norm_func.im2tensor(img_raw)
        img = self.norm_func.process_unlabel(wa_img)
        mask = self.random_mask(img).squeeze(0)
        return img, mask

    def _load_flow_for_sequence(self, ordered_list, start_idx):
        if not self.flow_root:
            return None
        idxs = self._get_sequence_indices_from_list(ordered_list, start_idx)
        flows = []
        for i in range(len(idxs) - 1):
            f0 = ordered_list[idxs[i]]
            flow_fn = os.path.join(self.flow_root, f"{f0}_flow{self.flow_ext}")
            if os.path.exists(flow_fn):
                if self.flow_ext == '.npy':
                    arr = np.load(flow_fn, mmap_mode='r')
                    flows.append(torch.from_numpy(np.asarray(arr)).float())
                else:
                    obj = torch.load(flow_fn, map_location='cpu')
                    flows.append(obj if isinstance(obj, torch.Tensor) else torch.from_numpy(np.asarray(obj)).float())
            else:
                flows.append(None)
        if len(flows) == 0:
            return None
        # find first non-none to infer shape
        first = next((f for f in flows if f is not None), None)
        if first is None:
            return None
        stacked = []
        for f in flows:
            if f is None:
                stacked.append(torch.zeros_like(first))
            else:
                stacked.append(f)
        return torch.stack(stacked, dim=0)  # (T-1, 2, H, W)

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

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def __call__(self, tensor):
        mean = self.mean.to(tensor.device).view(3, 1, 1)
        std = self.std.to(tensor.device).view(3, 1, 1)
        return tensor * std + mean

if __name__ == '__main__':
    datadir = "/qnap/home_archive/wlin38/crowd/data/ori_data/ShanghaiTech/part_A"
    protc_path = '../../dac_label/sha-5.txt'
    data = SHHA(datadir, 'train', protc_path=protc_path, seq_len=3, seq_stride=1)

    denormal = DeNormalize(
         mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    import cv2

    def cv2fig(tensor, name=None):
        # print(name, "", tensor.shape)
        limg = denormal(tensor).squeeze(0)
        limg = limg.permute(1, 2, 0).cpu().numpy()
        limg = (limg * 255).astype(np.uint8)
        limg = cv2.cvtColor(limg, cv2.COLOR_RGB2BGR)
        if name is not None:
            cv2.imwrite(name, limg)
        return limg
        
    
    from torch.utils.data import DataLoader

    loader = DataLoader( data, batch_size = 2, num_workers = 1, pin_memory=False, shuffle = True, collate_fn=SHHA.collate_fn)    
    for i, batch in enumerate(tqdm.tqdm(loader)):
        print([x.shape if hasattr(x,'shape') else type(x) for x in batch])
        break
