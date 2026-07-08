# -*- coding: utf-8 -*-
"""
preprocess/preprocess_fdst.py

Full preprocessing pipeline for FDST-style dataset layout.
- Converts VIA-like JSON annotations (per-image .json) into GT_*.npy (Nx2 center points) under new-anno/
- Computes dense optical flow for adjacent frames (Farneback baseline) and resamples flow to decoder output resolution (H_out=W_img//down)
- Saves flows under flow_root as {video}_{frame}_flow.npy (per-video) or {frame}_flow.npy (flat)
- Safe, idempotent: will skip existing outputs unless --overwrite is passed

Usage example:
python preprocess/preprocess_fdst.py \
  --root /media/SSD/OSshareSpace/localSpace/workPlace/zhongKe/gd/FDST \
  --modes train test \
  --down 4 \
  --ext jpg \
  --compute-flow \
  --flow-out /media/SSD/OSshareSpace/localSpace/workPlace/zhongKe/gd/FDST/precomputed_flow

This script is intended to be run once per dataset preparation. It handles both per-video (images/<video>/frame.jpg)
and flat (images/frame.jpg) layouts.
"""
#python preprocess/preprocess_fdst.py --root /media/SSD/OSshareSpace/localSpace/workPlace/zhongKe/gd/FDST --modes train --ext jpg
import os
import argparse
import json
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


def parse_via_json_file(json_path):
    """
    Parse a VIA-like JSON file and return Nx2 numpy array of (cx, cy) center points.
    If multiple entries present in file, collect regions across them.
    """
    try:
        with open(json_path, 'r') as f:
            jd = json.load(f)
    except Exception:
        return np.zeros((0, 2), dtype=np.float32)

    pts = []
    if isinstance(jd, dict):
        # VIA exports are often dicts mapping keys -> entry
        for key, entry in jd.items():
            regions = entry.get('regions', [])
            for r in regions:
                sa = r.get('shape_attributes', {})
                name = sa.get('name')
                if name == 'rect':
                    x = sa.get('x'); y = sa.get('y'); w = sa.get('width'); h = sa.get('height')
                    if None in (x, y, w, h):
                        continue
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    pts.append([cx, cy])
                elif name in ('point',):
                    px = sa.get('cx') or sa.get('x')
                    py = sa.get('cy') or sa.get('y')
                    if px is None or py is None:
                        continue
                    pts.append([float(px), float(py)])
    # else: unsupported format
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def convert_jsons_to_npy(root, mode, ext='jpg', overwrite=False):
    """
    Convert per-image .json annotations to new-anno/GT_{video}_{frame}.npy or GT_{frame}.npy.
    Returns a list of produced npy paths.
    """
    produced = []
    images_dir = os.path.join(root, f"{mode}_data", 'images')
    anno_out = os.path.join(root, f"{mode}_data", 'new-anno')
    os.makedirs(anno_out, exist_ok=True)

    if not os.path.exists(images_dir):
        print(f"images dir not found: {images_dir}")
        return produced

    entries = sorted([e for e in os.listdir(images_dir) if not e.startswith('.')])
    has_subdirs = any(os.path.isdir(os.path.join(images_dir, e)) for e in entries)

    if has_subdirs:
        # per-video
        for vid in entries:
            vid_dir = os.path.join(images_dir, vid)
            if not os.path.isdir(vid_dir):
                continue
            frames = sorted([f for f in os.listdir(vid_dir) if f.lower().endswith(ext)])
            for fname in frames:
                name, _ = os.path.splitext(fname)
                # prefer .npy if exists already in annotations
                outp = os.path.join(anno_out, f"GT_{vid}_{name}.npy")
                if os.path.exists(outp) and not overwrite:
                    continue
                # try local jsons (same folder as image)
                json_candidates = [
                    os.path.join(vid_dir, f"{name}.json"),
                    os.path.join(vid_dir, f"{name}.jpg.json"),
                    os.path.join(root, f"{mode}_data", 'annotations', vid, f"{name}.json"),
                    os.path.join(root, f"{mode}_data", 'annotations', f"{name}.json"),
                    os.path.join(anno_out, f"{name}.json"),
                ]
                jp = next((p for p in json_candidates if p and os.path.exists(p)), None)
                if jp:
                    pts = parse_via_json_file(jp)
                else:
                    # if no json, check for existing npy with common names
                    alt_npy = None
                    possible_npy = [
                        os.path.join(root, f"{mode}_data", 'annotations', vid, f"{name}.npy"),
                        os.path.join(root, f"{mode}_data", 'new-anno', f"GT_{vid}_{name}.npy"),
                    ]
                    alt_npy = next((p for p in possible_npy if os.path.exists(p)), None)
                    if alt_npy:
                        try:
                            pts = np.load(alt_npy).astype(np.float32)
                        except Exception:
                            pts = np.zeros((0, 2), dtype=np.float32)
                    else:
                        pts = np.zeros((0, 2), dtype=np.float32)
                np.save(outp, pts)
                produced.append(outp)
    else:
        # flat
        frames = sorted([f for f in entries if f.lower().endswith(ext)])
        for fname in frames:
            name, _ = os.path.splitext(fname)
            outp = os.path.join(anno_out, f"GT_{name}.npy")
            if os.path.exists(outp) and not overwrite:
                continue
            json_candidates = [
                os.path.join(images_dir, f"{name}.json"),
                os.path.join(images_dir, f"{name}.jpg.json"),
                os.path.join(root, f"{mode}_data", 'annotations', f"{name}.json"),
                os.path.join(anno_out, f"{name}.json"),
            ]
            jp = next((p for p in json_candidates if p and os.path.exists(p)), None)
            if jp:
                pts = parse_via_json_file(jp)
            else:
                alt_npy = next((p for p in [os.path.join(root, f"{mode}_data", 'annotations', f"{name}.npy"), os.path.join(anno_out, f"GT_{name}.npy") ] if os.path.exists(p)), None)
                if alt_npy:
                    try:
                        pts = np.load(alt_npy).astype(np.float32)
                    except Exception:
                        pts = np.zeros((0, 2), dtype=np.float32)
                else:
                    pts = np.zeros((0, 2), dtype=np.float32)
            np.save(outp, pts)
            produced.append(outp)

    return produced


# Flow helper functions (Farneback baseline)
def compute_farneback(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    return np.transpose(flow, (2, 0, 1)).astype(np.float32)


def resample_flow(flow_chw, H_out, W_out):
    H_src, W_src = flow_chw.shape[1], flow_chw.shape[2]
    if H_src == H_out and W_src == W_out:
        return flow_chw.astype(np.float32)
    scale_x = float(W_out) / float(W_src)
    scale_y = float(H_out) / float(H_src)
    dx = cv2.resize(flow_chw[0], (W_out, H_out), interpolation=cv2.INTER_LINEAR) * scale_x
    dy = cv2.resize(flow_chw[1], (W_out, H_out), interpolation=cv2.INTER_LINEAR) * scale_y
    return np.stack([dx, dy], axis=0).astype(np.float32)


def compute_and_save_flow(root, mode, flow_out, down=4, ext='jpg', overwrite=False):
    """
    Compute Farneback flow for adjacent frames and save resampled flow to flow_out.
    """
    images_dir = os.path.join(root, f"{mode}_data", 'images')
    if not os.path.exists(images_dir):
        print(f"images dir not found: {images_dir}")
        return []
    os.makedirs(flow_out, exist_ok=True)
    produced = []

    entries = sorted([e for e in os.listdir(images_dir) if not e.startswith('.')])
    has_subdirs = any(os.path.isdir(os.path.join(images_dir, e)) for e in entries)

    if has_subdirs:
        for vid in entries:
            vid_dir = os.path.join(images_dir, vid)
            if not os.path.isdir(vid_dir):
                continue
            frames = sorted([f for f in os.listdir(vid_dir) if f.lower().endswith(ext)])
            for i in range(len(frames) - 1):
                f0 = frames[i]
                name0, _ = os.path.splitext(f0)
                out_fn = os.path.join(flow_out, f"{vid}_{name0}_flow.npy")
                if os.path.exists(out_fn) and not overwrite:
                    continue
                p0 = os.path.join(vid_dir, f0)
                p1 = os.path.join(vid_dir, frames[i + 1])
                prev = cv2.imread(p0)
                curr = cv2.imread(p1)
                if prev is None or curr is None:
                    continue
                prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                flow = compute_farneback(prev_gray, curr_gray)
                H_img, W_img = prev.shape[0], prev.shape[1]
                H_out = max(1, H_img // down)
                W_out = max(1, W_img // down)
                flow_rs = resample_flow(flow, H_out, W_out)
                np.save(out_fn, flow_rs)
                produced.append(out_fn)
    else:
        frames = sorted([f for f in entries if f.lower().endswith(ext)])
        for i in range(len(frames) - 1):
            f0 = frames[i]
            name0, _ = os.path.splitext(f0)
            out_fn = os.path.join(flow_out, f"{name0}_flow.npy")
            if os.path.exists(out_fn) and not overwrite:
                continue
            p0 = os.path.join(images_dir, f0)
            p1 = os.path.join(images_dir, frames[i + 1])
            prev = cv2.imread(p0)
            curr = cv2.imread(p1)
            if prev is None or curr is None:
                continue
            prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            flow = compute_farneback(prev_gray, curr_gray)
            H_img, W_img = prev.shape[0], prev.shape[1]
            H_out = max(1, H_img // down)
            W_out = max(1, W_img // down)
            flow_rs = resample_flow(flow, H_out, W_out)
            np.save(out_fn, flow_rs)
            produced.append(out_fn)

    return produced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='FDST root dir')
    parser.add_argument('--modes', nargs='+', default=['train', 'test'], help='modes to process')
    parser.add_argument('--ext', default='jpg', help='image extension (jpg/png)')
    parser.add_argument('--down', type=int, default=4, help='downsampling factor to produce flow resolution')
    parser.add_argument('--compute-flow', action='store_true', help='whether to compute flow')
    parser.add_argument('--flow-out', default=None, help='where to store computed flow; if None, uses <root>/precomputed_flow')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing outputs')
    args = parser.parse_args()

    root = args.root
    for mode in args.modes:
        print(f"Processing mode={mode} ...")
        npys = convert_jsons_to_npy(root, mode, ext=args.ext, overwrite=args.overwrite)
        print(f"Wrote {len(npys)} annotation npy files for mode={mode}")
        if args.compute_flow:
            flow_out = args.flow_out or os.path.join(root, 'precomputed_flow')
            flows = compute_and_save_flow(root, mode, flow_out, down=args.down, ext='.' + args.ext if not args.ext.startswith('.') else args.ext, overwrite=args.overwrite)
            print(f"Wrote {len(flows)} flow files for mode={mode} into {flow_out}")

    print('Preprocessing finished')

if __name__ == '__main__':
    main()
