# -*- coding: utf-8 -*-
"""
preprocess/compute_and_resample_flow.py

Compute optical flow (Farneback baseline) for adjacent frames and resample to decoder output resolution.
Saves flow as (2, H_out, W_out) float32 numpy files named either {video}_{frame}_flow.npy or {frame}_flow.npy
"""

import os
import argparse
from glob import glob
import cv2
import numpy as np


def compute_farneback(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    return np.transpose(flow, (2,0,1)).astype(np.float32)


def resample_flow(flow_chw, H_out, W_out):
    H_src, W_src = flow_chw.shape[1], flow_chw.shape[2]
    scale_x = float(W_out) / float(W_src)
    scale_y = float(H_out) / float(H_src)
    dx = cv2.resize(flow_chw[0], (W_out, H_out), interpolation=cv2.INTER_LINEAR) * scale_x
    dy = cv2.resize(flow_chw[1], (W_out, H_out), interpolation=cv2.INTER_LINEAR) * scale_y
    return np.stack([dx, dy], axis=0).astype(np.float32)


def process_folder(img_folder, out_root, down, ext='.jpg'):
    if any(os.path.isdir(os.path.join(img_folder, d)) for d in os.listdir(img_folder)):
        vids = sorted([d for d in os.listdir(img_folder) if os.path.isdir(os.path.join(img_folder, d))])
        for vid in vids:
            vid_dir = os.path.join(img_folder, vid)
            frames = sorted([f for f in os.listdir(vid_dir) if f.endswith(ext)])
            for i in range(len(frames)-1):
                f0, f1 = frames[i], frames[i+1]
                p0 = os.path.join(vid_dir, f0); p1 = os.path.join(vid_dir, f1)
                out_fn = os.path.join(out_root, f"{vid}_{os.path.splitext(f0)[0]}_flow.npy")
                if os.path.exists(out_fn):
                    continue
                prev = cv2.imread(p0)
                curr = cv2.imread(p1)
                if prev is None or curr is None:
                    continue
                prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                flow = compute_farneback(prev_gray, curr_gray)
                H_img, W_img = prev.shape[0], prev.shape[1]
                H_out = H_img // down; W_out = W_img // down
                flow_rs = resample_flow(flow, H_out, W_out)
                os.makedirs(out_root, exist_ok=True)
                np.save(out_fn, flow_rs)
    else:
        frames = sorted([f for f in os.listdir(img_folder) if f.endswith(ext)])
        for i in range(len(frames)-1):
            f0, f1 = frames[i], frames[i+1]
            p0 = os.path.join(img_folder, f0); p1 = os.path.join(img_folder, f1)
            out_fn = os.path.join(out_root, f"{os.path.splitext(f0)[0]}_flow.npy")
            if os.path.exists(out_fn):
                continue
            prev = cv2.imread(p0)
            curr = cv2.imread(p1)
            if prev is None or curr is None:
                continue
            prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            flow = compute_farneback(prev_gray, curr_gray)
            H_img, W_img = prev.shape[0], prev.shape[1]
            H_out = H_img // down; W_out = W_img // down
            flow_rs = resample_flow(flow, H_out, W_out)
            os.makedirs(out_root, exist_ok=True)
            np.save(out_fn, flow_rs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root', required=True)
    parser.add_argument('--out_root', required=True)
    parser.add_argument('--down', type=int, default=4)
    parser.add_argument('--ext', default='.jpg')
    args = parser.parse_args()
    process_folder(args.img_root, args.out_root, args.down, ext=args.ext)
