# FDST 時序人群計數訓練 — 代碼總結

## 整體文件結構

```
P2RLoss/
├── config.py                          # yacs 配置定義（含 SEQ_LEN / FLOW_ROOT 等）
├── main.py                            # 訓練入口：兩階段訓練 + validate
├── datasets/
│   ├── __init__.py                    # build_loader：根據 config 建立 FDST DataLoader
│   ├── fdst.py                        # FDST dataset：影片序列讀取、光流加載、隨機遮罩
│   └── utils.py                       # NormalSample：資料增強、裁剪、歸一化
├── models/
│   ├── vgg16bn.py                     # VGG16BN + ConvGRU 時序模型
│   └── utils.py                       # UpSample_P2P, SimpleDecoder, ConvGRUCell, TemporalUnit
├── losses/
│   └── p2rloss.py                     # Point-to-Region Loss (P2RLoss)
├── train/
│   └── train_loop.py                  # 訓練循環：supervised + semi-supervised
└── preprocess/
    └── preprocess_fdst.py             # 預處理：VIA JSON → GT_*.npy + Farneback 光流計算
```

---

## 1. 配置定義 — `config.py`

```python
# 時序訓練相關配置（新增於 DATA 區塊）
_C.DATA.SEQ_LEN = 1          # 序列長度（幀數）
_C.DATA.SEQ_STRIDE = 1       # 幀間步長
_C.DATA.FLOW_ROOT = ''       # 預計算光流文件根目錄
_C.DATA.FLOW_EXT = '.npy'    # 光流文件副檔名

# 透過 --opts 可覆蓋：
# --opts DATA.DATASET fdst DATA.SEQ_LEN 2 DATA.SEQ_STRIDE 1 \
#        DATA.FLOW_ROOT /path/to/flow DATA.FLOW_EXT .npy
```

---

## 2. 訓練入口 — `main.py`

### 2.1 兩階段訓練策略

```python
STAGE_1 = 25       # 第 0-24 epoch：純 supervised
STAGE_2 = 50       # 第 25-49 epoch：半監督（supervised + 時序 pseudo-label）
                   # 第 50-1499 epoch：繼續半監督

for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
    stage = 'sup' if epoch < STAGE_1 else 'semi'

    if stage == 'semi' and epoch == STAGE_1:
        teacher.load_state_dict(student.state_dict())  # Teacher ← pre-trained student
```

### 2.2 模型初始化

```python
student, teacher = build_model(config.MODEL)
teacher = copy.deepcopy(student)
for p in teacher.parameters():
    p.requires_grad = False
```

### 2.3 驗證函數（每 SAVE_FREQ=5 epoch）

```python
def validate(config, data_loader, model, criterion):
    for idx, batch in enumerate(data_loader):
        images, dotseq, imgid = batch[0], batch[1], batch[2]
        if images.dim() == 5:          # (B, T, C, H, W) → 取第一幀
            images = images[:, 0]
        output, _ = model(images, prev_h=None)
        loss = criterion(output, dotseq, down).item()
        outnum = (output > 0).sum(dim=(1, 2, 3))  # 計數估計
        mae, mse = |outnum - cnt|, (diff^2).mean()
```

---

## 3. 訓練循環 — `train/train_loop.py`

### 3.1 有標註監督損失

```python
# 取 labeled 序列的第一幀計算 supervised P2R loss
sup_loss = 0.0
if limg_batch is not None and limg_batch.size(0) > 0:
    lframe = limg_batch[:, 0]           # (B, C, H, W)
    lpred, _ = student(lframe, prev_h=None)
    ldots = [seq[0][0][:, :2].to(device) for seq in lseqs]  # ground truth dots
    sup_loss = p2r(lpred, ldots, down=down)
```

### 3.2 Stage 1（純監督）

```python
if stage == 'sup':
    optimizer.zero_grad()
    scaler.scale(sup_loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 3.3 Stage 2（半監督 + 時序 pseudo-label）

```python
elif stage == 'semi':
    # 初始化 ConvGRU hidden states
    student_h = student.init_hidden(B, device, (H//down, W//down))
    teacher_h = teacher.init_hidden(B, device, (H//down, W//down))

    # Teacher 在 t=0 的預測
    with torch.no_grad():
        teacher_pred0, teacher_h = teacher(uimg_batch[:, 0], teacher_h)
        teacher_pred_prev = teacher_pred0.detach()

    for t in range(1, T):            # 時序循環：t=1 .. T-1
        optimizer.zero_grad()

        # Student forward on frame t
        student_pred_t, student_h = student(uimg_batch[:, t], student_h)

        # Teacher 從 t-1 提取 pseudo-points，透過光流 warp 到 t
        with torch.no_grad():
            points_prev = generate_points_from_density(teacher_pred_prev)
            if uflows_batch is not None:
                warped = warp_points_with_flow(points_prev, uflows_batch[:, t-1])
            else:
                warped = points_prev

        # 無監督 P2R loss
        loss_p2r = p2r(student_pred_t, warped, down=down, masks=None)
        loss = loss_p2r + sup_loss / (T - 1)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Truncated BPTT + EMA
        student_h = student_h.detach()
        update_ema(student, teacher, ema_alpha)

        # Teacher forward on frame t（為下一時間步做準備）
        with torch.no_grad():
            teacher_pred_t, teacher_h = teacher(uimg_batch[:, t], teacher_h)
            teacher_pred_prev = teacher_pred_t.detach()
```

### 3.4 輔助函數

**Pseudo-points 生成**（從 density logits 提取局部極大值點）：
```python
def generate_points_from_density(density_logits, threshold=0.3):
    prob = torch.sigmoid(density_logits)                # (B,1,H,W)
    pool = F.max_pool2d(prob, kernel_size=3, padding=1)
    peaks = (prob == pool) & (prob > threshold)          # 局部極大值 + 閾值
    # 返回 list of (N_i, 2) in (x, y) 座標
```

**光流採樣**（在點位置採樣 flow vector）：
```python
def sample_flow_at_points(flow, points):
    # flow: (B, 2, H, W), points: (B, N, 2) in (x, y) pixel coords
    nx = (xs / (W-1)) * 2 - 1          # 歸一化至 [-1, 1]
    ny = (ys / (H-1)) * 2 - 1
    sampled = F.grid_sample(f, grid)   # 雙線性採樣
    return sampled                     # (B, N, 2) flow vectors
```

**光流 Warp**：
```python
def warp_points_with_flow(points_list, flow_batch):
    # points_list[b] = (N_i, 2), flow_batch = (B, 2, H, W)
    sampled = sample_flow_at_points(flow_batch, padded_points)
    warped_pts = points_list[b] + flow_vecs              # 點 + flow = warp 後位置
    return warped_list
```

**EMA 更新**：
```python
def update_ema(student, teacher, alpha=0.999):
    for sp, tp in zip(student.parameters(), teacher.parameters()):
        tp.data.mul_(alpha).add_(sp.data, alpha=1.0 - alpha)
```

---

## 4. FDST 資料集 — `datasets/fdst.py`

### 4.1 目錄掃描與索引建立

```python
class FDST(data.Dataset):
    def __init__(self, root_path, mode, label_prob, protc_path,
                 seq_len=1, seq_stride=1, flow_root=None, flow_ext='.npy'):
        images_dir = f"{root_path}/{mode}_data/images"

        # 自動檢測子目錄（video_id）或扁平結構
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
                self.label.append((vid, frm))    # 有標註
            self.unlabel.append((vid, frm))      # 所有幀皆為 unlabeled
```

### 4.2 `__getitem__` — 返回序列

```python
def __getitem__(self, index):
    if self.training:
        # labeled seq: 從 label 集隨機取一幀，往前取 seq_len 張
        lid = random.choice(self.label)
        limg_seq, lseqs = self.readLabelSequenceFromTuple(lid)
        # unlabeled seq: 按 index 取
        uid = self.unlabel[index % len(self.unlabel)]
        uimg_seq, umask_seq = self.readUnlabelSequenceFromTuple(uid)
        # 光流: 可選
        uflows = self._load_flow_for_sequence(self.unlabel, ...)
        return limg_seq, lseqs, lid, uimg_seq, umask_seq, uid, uflows
    else:
        return limg_seq, lseqs, lid, flows
```

### 4.3 序列索引生成

```python
def _get_sequence_indices(self, list_ref, start_idx):
    seq_idx = []
    for i in range(self.seq_len):
        idx = start_idx + i * self.seq_stride
        idx = min(idx, len(list_ref) - 1)      # 邊界 clamp
        seq_idx.append(idx)
    return seq_idx     # 例如 [42, 43, 44]（seq_len=3, stride=1）
```

### 4.4 有標註序列讀取

```python
def readLabelSequenceFromTuple(self, tpl):
    idx = self.label.index(tpl)
    idxs = self._get_sequence_indices(self.label, idx)
    for ii in idxs:
        vid, frm = self.label[ii]
        img = self._load_image(vid, frm)            # 載入圖片
        img_t = self.norm_func.im2tensor(img)        # ToTensor + Normalize
        dot = self._load_annotation(vid, frm)        # 載入點標註 (N,2)
        img_t, dot = self.norm_func.process_lable(img_t, dot)  # 裁剪 + 增強
        imgs.append(img_t.squeeze(0))                # (3, H, W)
        dotseqs.append(dot)                          # [tensor(N,3)]
    imgs_seq = torch.stack(imgs, dim=0)              # (T, 3, H, W)
    return imgs_seq, dotseqs
```

### 4.5 無標註序列讀取

```python
def readUnlabelSequenceFromTuple(self, tpl):
    for ii in idxs:
        img = self._load_image(vid, frm)
        wa_img = self.norm_func.im2tensor(img)       # 弱增強
        img_proc = self.norm_func.process_unlabel(wa_img)  # 隨機裁剪
        imgs.append(img_proc.squeeze(0))             # (3, H, W)
        masks.append(self.random_mask(img_proc).squeeze(0))  # 隨機遮罩
    imgs_seq = torch.stack(imgs, dim=0)              # (T, 3, H, W)
    masks_seq = torch.stack(masks, dim=0)            # (T, 1, H, W)
    return imgs_seq, masks_seq
```

### 4.6 光流加載

```python
def _load_flow_for_sequence(self, list_ref, start_idx):
    if not self.flow_root:
        return None
    for i in range(seq_len - 1):                     # T-1 個 flow 幀對
        cand1 = f"{flow_root}/{vid}_{frm}_flow{ext}"  # 影片子目錄命名
        cand2 = f"{flow_root}/{frm}_flow{ext}"         # 扁平命名
        arr = np.load(fn)                              # shape: (2, H, W)
        flows.append(torch.from_numpy(arr))
    return torch.stack(flows, dim=0)                   # (T-1, 2, H, W)
```

### 4.7 collate_fn — 批次組裝

```python
@staticmethod
def collate_fn(samples):
    # 訓練: 7 items
    limg_batch = torch.stack(limg_seqs)     # (B, T, 3, H, W)
    uimg_batch = torch.stack(uimg_seqs)     # (B, T, 3, H, W)
    umask_batch = torch.stack(umask_seqs)   # (B, T, 1, H, W)
    uflows_batch = torch.stack(flows)       # (B, T-1, 2, H, W) 或 None
    return limg_batch, lseqs, lids, uimg_batch, umask_batch, uids, uflows_batch
```

### 4.8 隨機遮罩

```python
def random_mask(self, uimgs):
    # 對每張圖生成一個 1/8 ~ 1/4 大小的矩形遮罩區域
    cut_w = img_w * random(1/8, 1/4)
    cut_h = img_h * random(1/8, 1/4)
    mask[:, :, top:bottom, left:right] = 0   # 遮罩區域設為 0
    return mask                              # (B, 1, H, W)
```

### 4.9 標註檔案查找

```python
def _load_annotation(self, vid, frm):
    candidates = [
        "new-anno/GT_{vid}_{frm}.npy",
        "new-anno/GT_{frm}.npy",
        "annotations/{vid}/{frm}.npy",
        "annotations/{frm}.npy",
    ]
    return torch.from_numpy(np.load(p))[:, :2]  # (N, 2) in (x, y)
```

---

## 5. 資料增強 — `datasets/utils.py`

### 5.1 NormalSample

```python
class NormalSample:
    im2tensor = Compose([ToTensor(), Normalize(mean, std)])

    strong_aug = Compose([
        ColorJitter(0.4, 0.4, 0.4, 0.1),    # p=0.8
        RandomGrayscale(p=0.25),
        GaussianBlur([.1, 2.]),              # p=0.8
        im2tensor
    ])
```

### 5.2 有標註處理

```python
def process_lable(self, image, dotseq):
    images, dotseqs = self.crop_and_resize(image, dotseq)  # 隨機裁剪 + resize
    images = F.pad(images, to_32_alignment)                 # 32 對齊
    images, dotseqs = random_horizontal_flip(images, dotseqs)
    for i, seq in enumerate(dotseqs):
        u = self.nearest(seq)                              # 最近鄰距離
        dotseqs[i] = torch.cat((seq[:, [1, 0]], u), dim=1)  # (y, x, nn_dist)
    return images, dotseqs                                   # (1,3,256,256), [tensor(N,3)]
```

### 5.3 隨機裁剪與縮放

```python
def crop_and_resize(self, image, dotseq=None, num_patches=1):
    scale = random() * 0.6 + 0.7           # 縮放 0.7 ~ 1.3
    crop_h = 256 / scale
    crop_w = 256 / scale
    # 隨機裁剪
    start_h, start_w = randint(0, imh - crop_h), randint(0, imw - crop_w)
    crop_img = image[:, start_h:end_h, start_w:end_w]
    crop_img = F.interpolate(crop_img, (256, 256))           # resize to 256x256
    if dotseq is not None:
        idx = (dotseq[:,0] >= start_w) & ... & (dotseq[:,1] <= end_h)
        selected_dot = dotseq[idx]
        selected_dot[:, 0] = (selected_dot[:, 0] - start_w) * (256/crop_w)
        selected_dot[:, 1] = (selected_dot[:, 1] - start_h) * (256/crop_h)
    return crop_imgs, crop_dots    # (1,3,256,256), [tensor(N',2)]
```

---

## 6. DataLoader 建立 — `datasets/__init__.py`

```python
def build_loader(config, mode):
    seq_len = getattr(config.DATA, 'SEQ_LEN', 1)
    seq_stride = getattr(config.DATA, 'SEQ_STRIDE', 1)
    flow_root = getattr(config.DATA, 'FLOW_ROOT', None)
    flow_ext = getattr(config.DATA, 'FLOW_EXT', '.npy')

    Dataset = {'shha': SHHA, 'fdst': FDST}[config.DATASET]
    data_set = Dataset(data_path, mode, label_prob, protc_path,
                       seq_len=seq_len, seq_stride=seq_stride,
                       flow_root=flow_root, flow_ext=flow_ext)
    return DataLoader(data_set,
                      batch_size=config.BATCH_SIZE,  # train: config value
                      shuffle=(mode == 'train'),
                      collate_fn=Dataset.collate_fn)
```

---

## 7. VGG16BN + ConvGRU 模型 — `models/vgg16bn.py`

### 7.1 模型結構

```python
class VGG16_BN(nn.Module):
    def __init__(self):
        # 編碼器: VGG16BN 特徵提取 (至第 33, 43 層)
        self.encoders = ModuleList([
            Sequential(*features[0:33]),   # 輸出 512ch
            Sequential(*features[33:43]),  # 輸出 512ch
        ])
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
        fea2 = self.encoding(image)                    # 編碼
        denmap = self.decoding(fea2)                   # 解碼 → density (B,1,H/4,W/4)
        x_t = self.to_temporal(denmap)                 # 投影
        next_h = self.temporal(x_t, prev_h)            # ConvGRU
        pred_logits = self.hidden2logit(next_h)        # 輸出 logits (B,1,H/4,W/4)
        return pred_logits, next_h
```

### 7.2 Hidden State 初始化

```python
def init_hidden(self, batch_size, device, spatial_size):
    return torch.zeros(batch_size, self.temporal_hidden, H, W, device=device)
```

---

## 8. ConvGRU 實現 — `models/utils.py`

### 8.1 ConvGRU Cell

```python
class ConvGRUCell(nn.Module):
    def __init__(self, in_ch=1, hid_ch=32, kernel_size=3):
        # z, r gates: conv(in_ch + hid_ch → 2*hid_ch)
        self.conv_zr = Conv2d(in_ch + hid_ch, 2 * hid_ch, 3, padding=1)
        # candidate: conv(in_ch + hid_ch → hid_ch)
        self.conv_n = Conv2d(in_ch + hid_ch, hid_ch, 3, padding=1)

    def forward(self, x, h):
        if h is None:
            h = torch.zeros(batch, hid_ch, H, W)
        cat = torch.cat([x, h], dim=1)
        z, r = torch.sigmoid(self.conv_zr(cat).chunk(2, dim=1))
        cat_r = torch.cat([x, r * h], dim=1)
        n = torch.tanh(self.conv_n(cat_r))
        h_next = (1 - z) * n + z * h
        return h_next
```

### 8.2 TemporalUnit Wrapper

```python
class TemporalUnit(nn.Module):
    def forward(self, x, h):
        return self.cell(x, h)   # 目前僅支援 'convgru' 模式
```

---

## 9. P2R Loss — `losses/p2rloss.py`

```python
class P2RLoss(nn.modules.loss._Loss):
    def forward(self, dens, seqs, down, masks=None):
        for i in range(bs):
            den, seq = dens[i], seqs[i]         # 每樣本的 density + 點標註
            A_coord = meshgrid(H, W) * down     # 像素座標 → 原圖座標
            A = den.view(H*W, 1)                # density 值
            B_coord = seq[:, :2]                # ground truth 點座標

            C = |A_coord - B_coord|             # 成本矩陣 (L2 distance)

            # Optimal transport-like matching
            M = nearest_neighbor_mask(C)        # 每個點匹配最近 pixel
            C = C / max_radius                  # 歸一化
            C2 = M*C + (1-M)*(C.max()+1)        # 遮罩未匹配區域
            T = argmin(C2)                      # 每個 pixel 的目標指派

            loss = BCEWithLogits(A, T, weight)  # 加權二元交叉熵
        return loss.mean()
```

---

## 10. 光流預處理 — `preprocess/preprocess_fdst.py`

### 10.1 標註轉換（VIA JSON → GT_*.npy）

```python
def convert_jsons_to_npy(root, mode):
    for each image:
        json_path = find_json(images_dir, annotations_dir)
        pts = parse_via_json_file(json_path)   # VIA 格式解析（rect/point）
        np.save(f"new-anno/GT_{vid}_{frm}.npy", pts)   # (N, 2) float32
```

### 10.2 Farneback 光流計算

```python
def compute_farneback(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2)
    return flow.transpose(2, 0, 1).astype(np.float32)   # (2, H, W)
```

### 10.3 光流重採樣

```python
def resample_flow(flow_chw, H_out, W_out):
    # 將光流從原圖解析度 resize 到模型輸出解析度
    scale_x = W_out / flow_W
    scale_y = H_out / flow_H
    dx = cv2.resize(flow[0], (W_out, H_out)) * scale_x    # 位移量等比縮放
    dy = cv2.resize(flow[1], (W_out, H_out)) * scale_y
    return np.stack([dx, dy], axis=0).astype(np.float32)  # (2, H/4, W/4)
```

### 10.4 完整預處理流程

```python
def compute_and_save_flow(root, mode, flow_out, down=4):
    for vid_dir in videos:
        for i in range(len(frames) - 1):
            f0, f1 = frames[i], frames[i+1]
            prev, curr = cv2.imread(f0), cv2.imread(f1)
            flow = compute_farneback(prev_gray, curr_gray)     # (2, H, W)
            flow_rs = resample_flow(flow, H//down, W//down)    # (2, H/4, W/4)
            np.save(f"{flow_out}/{vid}_{f0}_flow.npy", flow_rs)
```

---

## 訓練命令

```bash
python main.py --data-path /path/to/FDST \
    --label 5 --protocol /path/to/protocol.txt \
    --batch-size 16 --tag fdst-experiment \
    --opts DATA.DATASET fdst \
           DATA.SEQ_LEN 2 DATA.SEQ_STRIDE 1 \
           DATA.FLOW_ROOT /path/to/FDST/precomputed_flow \
           TRAIN.EPOCHS 1500
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

  collate_fn → 7-tuple: (B,T,3,256), lseqs_list, lids, (B,T,3,256), (B,T,1,256), uids, (B,T-1,2,64,64) or None

  train_one_epoch(stage='sup')
    → supervised P2R loss on labeled first frame

  train_one_epoch(stage='semi')
    → supervised P2R loss + temporal P2R loss with flow-warped pseudo-labels
    → EMA teacher update + truncated BPTT

  validate()
    → MAE/MSE on (output > 0).sum() vs ground truth count
```
