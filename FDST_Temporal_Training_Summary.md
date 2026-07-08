# FDST 時序人群計數訓練 — 代碼總結

## 整體文件結構

```
P2RLoss/
├── config.py                          # yacs 配置定義（含 SEQ_LEN / FLOW_ROOT / NUM_WORKERS 等）
├── main.py                            # 訓練入口：兩階段訓練（sup→semi）+ validate
├── datasets/
│   ├── __init__.py                    # build_loader：DataLoader 建立（含優化參數）
│   ├── fdst.py                        # FDST dataset：影片序列讀取、光流加載、隨機遮罩
│   └── utils.py                       # NormalSample：資料增強、裁剪、歸一化、最近鄰
├── models/
│   ├── vgg16bn.py                     # VGG16BN + ConvGRU 時序模型
│   └── utils.py                       # UpSample_P2P, SimpleDecoder, ConvGRUCell, TemporalUnit
├── losses/
│   └── p2rloss.py                     # Point-to-Region Loss (P2RLoss) — 向量化實作
├── train/
│   └── train_loop.py                  # 訓練循環：supervised + semi-supervised + tqdm 進度條
└── preprocess/
    └── preprocess_fdst.py             # 預處理：VIA JSON → GT_*.npy + Farneback 光流計算 + 重採樣
```

---

## 1. 配置定義 — `config.py`

```python
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 1
_C.DATA.DATA_PATH = ''
_C.DATA.DATASET = 'shha'              # 'fdst' 使用 FDST
_C.DATA.PIN_MEMORY = False
_C.DATA.NUM_WORKERS = 4               # 資料載入 worker 數
_C.DATA.SEQ_LEN = 1                   # 序列長度（幀數）
_C.DATA.SEQ_STRIDE = 1                # 幀間步長
_C.DATA.FLOW_ROOT = ''                # 預計算光流文件根目錄
_C.DATA.FLOW_EXT = '.npy'             # 光流文件副檔名

# 模型
_C.MODEL.NAME = 'VGG16BN'
_C.MODEL.LOSS = 'P2R'

# 訓練
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 1500
_C.TRAIN.BASE_LR = 5e-5
_C.TRAIN.BACKBONE_LR = 1e-5
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 3500
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.9

# 透過 --opts 覆蓋範例：
# --opts DATA.DATASET fdst DATA.SEQ_LEN 2 DATA.SEQ_STRIDE 1 \
#        DATA.FLOW_ROOT /path/to/flow DATA.FLOW_EXT .npy \
#        DATA.NUM_WORKERS 4
```

---

## 2. DataLoader 建立 — `datasets/__init__.py`

```python
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')  # 避免 FD 耗盡

def build_loader(config, mode):
    Dataset = {'shha': SHHA, 'fdst': FDST}[config.DATASET.lower()]

    data_set = Dataset(data_path, mode, label_prob, protc_path,
                       seq_len=seq_len, seq_stride=seq_stride,
                       flow_root=flow_root, flow_ext=flow_ext)

    return DataLoader(
        data_set,
        batch_size=batch_size if mode == 'train' else 1,
        num_workers=num_workers,
        shuffle=(mode == 'train'),
        collate_fn=Dataset.collate_fn,
        persistent_workers=num_workers > 0,   # worker 跨 epoch 持續
        prefetch_factor=4 if num_workers > 0 else 2,  # 預取更多 batch
    )
```

---

## 3. FDST 資料集 — `datasets/fdst.py`

### 3.1 目錄掃描與索引建立

```python
class FDST(data.Dataset):
    def __init__(self, root_path, mode, label_prob, protc_path,
                 seq_len=1, seq_stride=1, flow_root=None, flow_ext='.npy'):
        images_dir = f"{root_path}/{mode}_data/images"

        # 自動檢測影片子目錄或扁平結構
        entries = os.listdir(images_dir)
        has_subdirs = any(os.path.isdir(os.path.join(images_dir, e)) for e in entries)

        self.frames = []  # [(video_id, frame_name), ...]
        if has_subdirs:
            for vid in sorted(dirs):
                for f in sorted(files_in_vid):
                    self.frames.append((vid, os.path.splitext(f)[0]))
        else:
            for f in sorted(files):
                self.frames.append((None, os.path.splitext(f)[0]))

        # 根據 protocol 文件決定 labeled frames
        idset = set(protocol_lines) if protc_path else None
        for vid, frm in self.frames:
            key = f"{vid}/{frm}" if vid else frm
            if (not self.training) or (idset is None) or (key in idset):
                self.label.append((vid, frm))     # 有標註
            self.unlabel.append((vid, frm))       # 所有幀皆為 unlabeled
```

### 3.2 `__getitem__` — 返回序列

```python
def __getitem__(self, index):
    if self.training:
        lid = random.choice(self.label)                          # 隨機有標註幀
        limg_seq, lseqs = self.readLabelSequenceFromTuple(lid)   # labeled 序列
        uid = self.unlabel[index % len(self.unlabel)]            # 按 index 取
        uimg_seq, umask_seq = self.readUnlabelSequenceFromTuple(uid)  # unlabeled 序列
        uflows = self._load_flow_for_sequence(self.unlabel, ...) # 可選光流
        return limg_seq, lseqs, lid, uimg_seq, umask_seq, uid, uflows
    else:
        return limg_seq, lseqs, lid, flows
```

### 3.3 序列索引生成

```python
def _get_sequence_indices(self, list_ref, start_idx):
    seq_idx = []
    for i in range(self.seq_len):
        idx = start + i * self.seq_stride
        idx = min(idx, len(list_ref) - 1)      # 邊界 clamp
        seq_idx.append(idx)
    return seq_idx     # 例如 seq_len=3, stride=1 → [42, 43, 44]
```

### 3.4 有標註序列讀取

```python
def readLabelSequenceFromTuple(self, tpl):
    idx = self.label.index(tpl)
    idxs = self._get_sequence_indices(self.label, idx)
    imgs, dotseqs = [], []
    for ii in idxs:
        vid, frm = self.label[ii]
        img = self._load_image(vid, frm)           # 載入圖片
        img_t = self.norm_func.im2tensor(img)       # ToTensor + Normalize
        dot = self._load_annotation(vid, frm)       # 載入點標註 (N,2)
        img_t, dot = self.norm_func.process_lable(img_t, dot)  # 裁剪 + 增強
        imgs.append(img_t.squeeze(0))               # (3, H, W)
        dotseqs.append(dot)                         # [tensor(N,3)]
    imgs_seq = torch.stack(imgs, dim=0)             # (T, 3, H, W)
    return imgs_seq, dotseqs
```

### 3.5 無標註序列讀取（已移除無用 strong_aug）

```python
def readUnlabelSequenceFromTuple(self, tpl):
    idx = self.unlabel.index(tpl)
    idxs = self._get_sequence_indices(self.unlabel, idx)
    imgs, masks = [], []
    for ii in idxs:
        vid, frm = self.unlabel[ii]
        img = self._load_image(vid, frm)
        wa_img = self.norm_func.im2tensor(img)
        img_proc = self.norm_func.process_unlabel(wa_img)    # 隨機裁剪 + pad
        imgs.append(img_proc.squeeze(0))
        masks.append(self.random_mask(img_proc).squeeze(0))  # 隨機遮罩
    imgs_seq = torch.stack(imgs, dim=0)     # (T, 3, H, W)
    masks_seq = torch.stack(masks, dim=0)   # (T, 1, H, W)
    return imgs_seq, masks_seq
```

### 3.6 圖片與標註載入（無緩存，避免記憶體爆炸）

```python
def _load_image(self, vid, frm):
    if vid is None:
        imgpath = os.path.join(self.img_dir, frm + '.jpg')
    else:
        imgpath = os.path.join(self.img_dir, vid, frm + '.jpg')
    if not os.path.exists(imgpath):  # fallback to .png
        ...
    return Image.open(imgpath).convert('RGB')

def _load_annotation(self, vid, frm):
    candidates = [
        "new-anno/GT_{vid}_{frm}.npy",
        "new-anno/GT_{frm}.npy",
        "annotations/{vid}/{frm}.npy",
        "annotations/{frm}.npy",
    ]
    return torch.from_numpy(np.load(p))[:, :2]   # (N, 2) in (x, y)
```

### 3.7 光流加載（無 mmap，直接讀取）

```python
def _load_flow_for_sequence(self, list_ref, start_idx):
    if not self.flow_root:
        return None
    for i in range(seq_len - 1):                     # T-1 個 flow 幀對
        cand1 = f"{flow_root}/{vid}_{frm}_flow{ext}"  # 影片子目錄命名
        cand2 = f"{flow_root}/{frm}_flow{ext}"         # 扁平命名
        arr = np.load(fn)                              # shape: (2, H, W)
        flows.append(torch.from_numpy(arr).float())
    return torch.stack(flows, dim=0)                   # (T-1, 2, H_out, W_out)
```

### 3.8 隨機遮罩

```python
def random_mask(self, uimgs):
    cut_img_mask = torch.ones((bsize, 1, img_h, img_w))
    for i in range(bsize):
        cut_w = int(img_w * random(1/8, 1/4))
        cut_h = int(img_h * random(1/8, 1/4))
        # 隨機位置矩形遮罩
        cut_img_mask[i, :, top:bottom, left:right] = 0
    return cut_img_mask   # (B, 1, H, W)
```

### 3.9 collate_fn — 批次組裝

```python
@staticmethod
def collate_fn(samples):
    # 訓練: 7 items
    limg_batch = torch.stack(limg_seqs)     # (B, T, C, H, W)
    uimg_batch = torch.stack(uimg_seqs)     # (B, T, C, H, W)
    umask_batch = torch.stack(umask_seqs)   # (B, T, 1, H, W)
    uflows_batch = torch.stack(flows) if any flow else None  # (B, T-1, 2, H, W)
    return limg_batch, lseqs, lids, uimg_batch, umask_batch, uids, uflows_batch

    # 評估: 4 items
    return limg_batch, lseqs, lids, flows_batch
```

---

## 4. 資料增強 — `datasets/utils.py`

### 4.1 NormalSample

```python
class NormalSample:
    im2tensor = Compose([ToTensor(), Normalize(mean, std)])

    # strong_aug 保留（預處理腳本用），訓練中不再調用以節省 CPU
    strong_aug = Compose([
        ColorJitter(0.4), RandomGrayscale(p=0.25),
        GaussianBlur([.1, 2.]), im2tensor
    ])
```

### 4.2 有標註處理

```python
def process_lable(self, image, dotseq):
    images, dotseqs = self.crop_and_resize(image, dotseq)   # 隨機裁剪 + resize
    images = F.pad(images, to_32_alignment)                  # 32 對齊
    images, dotseqs = random_horizontal_flip(images, dotseqs)
    for i, seq in enumerate(dotseqs):
        u = self.nearest(seq)                                # 最近鄰距離
        dotseqs[i] = torch.cat((seq[:, [1, 0]], u), dim=1)   # (y, x, nn_dist)
    return images, dotseqs                                   # (1,3,256), [tensor(N,3)]
```

### 4.3 隨機裁剪與縮放

```python
def crop_and_resize(self, image, dotseq=None, num_patches=1):
    scale = random() * 0.6 + 0.7              # 縮放 0.7 ~ 1.3
    crop_h, crop_w = 256/scale, 256/scale
    start_h, start_w = random_crop_position()
    crop_img = image[:, start_h:end_h, start_w:end_w]
    crop_img = F.interpolate(crop_img, (256, 256))   # bilinear resize
    if dotseq is not None:
        selected_dot = dotseq[idx]  # 篩選框內點 + 座標縮放
    return crop_imgs, crop_dots   # (1,3,256,256), [tensor(N',2)]
```

### 4.4 最近鄰距離（優化版：cdist + fill_diagonal）

```python
def nearest(self, seq):
    if seqlen <= 1:
        return torch.full((seqlen, 1), 32.0)
    dist2 = torch.cdist(seq, seq).pow(2)    # O(N^2) 距離矩陣
    dist2.fill_diagonal_(float('inf'))       # 排除自身
    m = dist2.min(dim=1).values              # 最近鄰（非自身）
    return m.view(-1, 1)
```

---

## 5. VGG16BN + ConvGRU 模型 — `models/vgg16bn.py`

```python
class VGG16_BN(nn.Module):
    def __init__(self):
        # 編碼器: VGG16BN 特徵提取至第 33, 43 層
        self.encoders = ModuleList([Sequential(features[0:33]),
                                    Sequential(features[33:43])])
        # 特徵融合
        self.fuse_layer = UpSample_P2P([512, 512], ouc=256)
        # 解碼器: 256ch → 1ch density map
        self.decoders = SimpleDecoder(256, 256, up_scale=2, out_channel=1)
        # ConvGRU 時序單元
        self.to_temporal = Conv2d(1, 1, kernel_size=1)
        self.temporal = TemporalUnit(mode='convgru', in_ch=1, hid_ch=32)
        self.hidden2logit = Conv2d(32, 1, kernel_size=1)
        self.down = 4   # 下採樣倍率

    def forward(self, image, prev_h=None):
        fea2 = self.encoding(image)                     # 編碼 (B,256,H/4,W/4)
        denmap = self.decoding(fea2)                    # 解碼 (B,1,H/4,W/4)
        x_t = self.to_temporal(denmap)                  # 投影
        next_h = self.temporal(x_t, prev_h)             # ConvGRU
        pred_logits = self.hidden2logit(next_h)         # 輸出 (B,1,H/4,W/4)
        return pred_logits, next_h

    def init_hidden(self, batch_size, device, spatial_size):
        return torch.zeros(batch_size, self.temporal_hidden, H, W, device=device)
```

---

## 6. ConvGRU 實現 — `models/utils.py`

```python
class ConvGRUCell(nn.Module):
    def __init__(self, in_ch=1, hid_ch=32, kernel_size=3):
        self.conv_zr = Conv2d(in_ch + hid_ch, 2 * hid_ch, 3, padding=1)
        self.conv_n  = Conv2d(in_ch + hid_ch, hid_ch, 3, padding=1)

    def forward(self, x, h):
        if h is None:
            h = torch.zeros(batch, hid_ch, H, W)
        cat = torch.cat([x, h], dim=1)
        z_chunks = self.conv_zr(cat).chunk(2, dim=1)
        z, r = torch.sigmoid(z_chunks[0]), torch.sigmoid(z_chunks[1])
        n = torch.tanh(self.conv_n(torch.cat([x, r * h], dim=1)))
        h_next = (1 - z) * n + z * h
        return h_next

class TemporalUnit(nn.Module):
    def forward(self, x, h):
        return self.cell(x, h)   # 目前僅支援 'convgru'
```

---

## 7. P2R Loss（向量化） — `losses/p2rloss.py`

```python
class P2RLoss(nn.modules.loss._Loss):
    def forward(self, dens, seqs, down, masks=None, crop_den_masks=None):
        # === Batch-wide grid coordinates（一次計算）===
        A_coord = meshgrid(H, W).view(1, 1, HW, 2) * down
        A = dens.view(bs, HW)

        # === 處理 empty / non-empty seqs ===
        T_full = zeros(bs, HW);  W_full = ones(bs, HW) * 0.5

        if nonempty_idx:
            # === Batched 距離矩陣（無 per-sample loop）===
            B_coord_batch = pad to max_N                 # (nb, 1, max_N, 2)
            C = |A_coord - B_coord_batch|                 # (nb, 1, HW, max_N)

            # === 批次匹配（scatter_ / argmin）===
            M = scatter_(argmin(C)) * (C < max_radis)
            maxC = clip((minC * M).amax(), min_radis, max_radis)
            C = C / maxC
            C = C * cost_point - A * cost_class

            vid = (M.sum(dim=2) > 0)
            C2 = M*C + (1-M)*(C.max()+1)
            T = scatter_(argmin(C2)) > 0.5               # 目標指派

            T_full[nonempty_idx] = T
            W_full[nonempty_idx] = T + 1

        # === 單次 BCE（一次 batch reduction）===
        loss = BCE_with_logits(A, T_full, weight=W_full, reduction='mean')
        return loss
```

---

## 8. 訓練循環 — `train/train_loop.py`

### 8.1 輔助函數

```python
def update_ema(student, teacher, alpha=0.999):
    for sp, tp in zip(student.parameters(), teacher.parameters()):
        tp.data.mul_(alpha).add_(sp.data, alpha=1.0 - alpha)

def generate_points_from_density(density_logits, threshold=0.3):
    prob = torch.sigmoid(density_logits)
    pool = F.max_pool2d(prob, kernel_size=3, padding=1)
    peaks = (prob == pool) & (prob > threshold)    # 局部極大值
    return list of (N_i, 2) in (x, y) per batch

def sample_flow_at_points(flow, points):
    # flow: (B, 2, H, W), points: (B, N, 2)
    nx = (xs / (W-1)) * 2 - 1                     # 歸一化至 [-1, 1]
    ny = (ys / (H-1)) * 2 - 1
    return F.grid_sample(f, grid)                  # 雙線性採樣 flow vector

def warp_points_with_flow(points_list, flow_batch):
    sampled = sample_flow_at_points(flow_batch, padded_points)
    warped_pts = points_list[b] + flow_vecs        # 點 + flow = warp 後位置
    return warped_list
```

### 8.2 主訓練函數

```python
def train_one_epoch(epoch, dataloader, student, teacher, optimizer, device,
                    down=4, ema_alpha=0.999, scaler=None, p2r_loss_cfg=None,
                    amp_enabled=True, min_batch_size=2, stage='sup'):

    pbar = tqdm(dataloader, desc=f'Epoch [{epoch}]', leave=False)
    for batch in pbar:
        limg_batch, lseqs, lids, uimg_batch, umask_batch, uids, uflows_batch = batch

        # === Supervised loss ===
        sup_loss = 0.0
        if limg_batch is not None:
            lframe = limg_batch[:, 0]
            lpred, _ = student(lframe, prev_h=None)
            ldots = [seq[0][0][:, :2].to(device) for seq in lseqs]
            sup_loss = p2r(lpred, ldots, down=down)

        if stage == 'sup':
            # === Stage 1: 純監督 ===
            scaler.scale(sup_loss).backward(); scaler.step(optimizer)

        elif stage == 'semi':
            # === Stage 2: 半監督 + 時序 pseudo-label ===
            student_h = student.init_hidden(B, device, (H//down, W//down))
            teacher_h = teacher.init_hidden(B, device, (H//down, W//down))

            # Teacher t=0 預測
            teacher_pred0, teacher_h = teacher(uimg_batch[:,0], teacher_h)

            for t in range(1, T):   # 時序循環（截斷 BPTT）
                optimizer.zero_grad()

                student_pred_t, student_h = student(uimg_batch[:,t], student_h)

                # Teacher pseudo-points + flow warp
                points_prev = generate_points_from_density(teacher_pred_prev)
                if uflows_batch is not None:
                    warped = warp_points_with_flow(points_prev, uflows_batch[:,t-1])
                loss_p2r = p2r(student_pred_t, warped, down=down)

                loss = loss_p2r + sup_loss / (T - 1)
                scaler.scale(loss).backward(); scaler.step(optimizer)

                student_h = student_h.detach()           # Truncated BPTT
                update_ema(student, teacher, ema_alpha)   # EMA 更新

                # Teacher forward for next step
                teacher_pred_t, teacher_h = teacher(uimg_batch[:,t], teacher_h)

        pbar.set_postfix(loss=f'{batch_loss:.4f}')
```

---

## 9. 訓練入口 — `main.py`

### 9.1 兩階段訓練策略

```python
STAGE_1 = 25        # Epoch 0-24:  純 supervised
STAGE_2 = 50        # Epoch 25-49: 半監督（teacher 從 supervised student 初始化）
                    # Epoch 50+:   繼續半監督

for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
    stage = 'sup' if epoch < STAGE_1 else 'semi'

    if stage == 'semi' and epoch == STAGE_1:
        teacher.load_state_dict(student.state_dict())   # Teacher ← pre-trained student

    temporal_train_one_epoch(..., stage=stage)
```

### 9.2 驗證（每 SAVE_FREQ=5 epoch）

```python
def validate(config, data_loader, model, criterion):
    for batch in data_loader:
        images, dotseq, imgid = batch[0], batch[1], batch[2]
        if images.dim() == 5:          # (B, T, C, H, W) → 取第一幀
            images = images[:, 0]
        output, _ = model(images, prev_h=None)
        loss = criterion(output, dotseq, down).item()
        outnum = (output > 0).sum(dim=(1,2,3))          # 計數估計
        mae, mse = |outnum - cnt|, (diff^2).mean()
```

---

## 10. 光流預處理 — `preprocess/preprocess_fdst.py`

### 10.1 Farneback 光流計算

```python
def compute_farneback(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2)
    return flow.transpose(2, 0, 1).astype(np.float32)   # (2, H, W)
```

### 10.2 光流重採樣（對齊模型輸出解析度）

```python
def resample_flow(flow_chw, H_out, W_out):
    scale_x = W_out / flow_W
    scale_y = H_out / flow_H
    dx = cv2.resize(flow[0], (W_out, H_out)) * scale_x    # 位移量等比縮放
    dy = cv2.resize(flow[1], (W_out, H_out)) * scale_y
    return np.stack([dx, dy], axis=0).astype(np.float32)  # (2, H/down, W/down)
```

---

## 訓練命令

```bash
python main.py --data-path /path/to/FDST \
    --label 5 --protocol /path/to/protocol.txt \
    --batch-size 16 --tag fdst-experiment \
    --opts DATA.DATASET fdst \
           DATA.SEQ_LEN 2 DATA.SEQ_STRIDE 1 \
           DATA.FLOW_ROOT /path/to/FDST/precomputed_flow
```

## 資料流總結

```
FDST 目錄結構:
  {root}/train_data/images/{video_id}/{frame}.jpg
  {root}/train_data/new-anno/GT_{video_id}_{frame}.npy      (N,2) 點標註
  {root}/precomputed_flow/{video_id}_{frame}_flow.npy         (2, H/4, W/4) 光流

訓練資料流:
  FDST.__getitem__(index)
    ├─ readLabelSequenceFromTuple()  →  (T,3,256,256), [tensor(N,3), ...]
    ├─ readUnlabelSequenceFromTuple() →  (T,3,256,256), (T,1,256,256)
    └─ _load_flow_for_sequence()     →  (T-1,2,64,64) or None

  collate_fn → 7-tuple: (B,T,3,256), lseqs_list, lids, (B,T,3,256),
                         (B,T,1,256), uids, (B,T-1,2,64,64) or None

  train_one_epoch(stage='sup')
    → supervised P2R loss on labeled first frame

  train_one_epoch(stage='semi')
    → supervised P2R loss + temporal P2R loss with flow-warped pseudo-labels
    → EMA teacher update + truncated BPTT

  validate()
    → MAE/MSE on (output > 0).sum() vs ground truth count
```
