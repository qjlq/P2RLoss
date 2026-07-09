# FDST 時序人群計數訓練 — 全套代碼與 Loss 不收斂分析

## 文件結構

```
P2RLoss/
├── config.py                     # yacs 配置 (NUM_WORKERS=4, SEQ_LEN=2, CHUNK_SIZE 等)
├── main.py                       # 入口: 兩階段訓練, validate, plot_curve
├── datasets/
│   ├── __init__.py               # DataLoader 建立 (file_descriptor, pin_memory, prefetch=4)
│   ├── fdst.py                   # FDST dataset: 序列讀取, 光流加載, O(1) dict, mmap
│   └── utils.py                  # NormalSample: crop_and_resize, pad, flip, 點座標轉換
├── models/
│   ├── vgg16bn.py                # VGG16BN + ConvGRU, init_hidden, down=4
│   └── utils.py                  # ConvGRUCell, TemporalUnit, SimpleDecoder, UpSample_P2P
├── losses/
│   └── p2rloss.py                # P2R Loss: 向量化 + chunked (CHUNK_SIZE=64), nearest on GPU
├── train/
│   └── train_loop.py             # 訓練循環: stage='sup' / 'semi', EMA, BPTT, flow warp, tqdm
└── preprocess/
    └── preprocess_fdst.py        # 離線: VIA JSON → GT_*.npy, Farneback 光流 + resample
```

---

## 1. 配置 — `config.py`

```python
_C.DATA.BATCH_SIZE = 1           # 可由 --batch-size 覆蓋
_C.DATA.DATASET = 'shha'         # 改為 'fdst'
_C.DATA.NUM_WORKERS = 4
_C.DATA.SEQ_LEN = 1
_C.DATA.SEQ_STRIDE = 1
_C.MODEL.LOSS = 'P2R'
_C.TRAIN.EPOCHS = 1500
_C.TRAIN.BASE_LR = 5e-5
_C.TRAIN.BACKBONE_LR = 1e-5      # backbone 專用 LR
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 3500   # StepLR 步長
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.9
```

**潛在問題**: `DECAY_EPOCHS=3500` > `EPOCHS=1500`, 所以 `StepLR` 從不衰減學習率。LR 始終為 `BASE_LR=5e-5` 和 `BACKBONE_LR=1e-5`。

---

## 2. DataLoader 建立 — `datasets/__init__.py`

```python
torch.multiprocessing.set_sharing_strategy('file_descriptor')
# ulimit increased to 65536

DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    pin_memory=True if mode == 'train' else False,
    shuffle=(mode == 'train'),
    collate_fn=Dataset.collate_fn,
    persistent_workers=num_workers > 0,
    prefetch_factor=4 if num_workers > 0 else 2,
)
```

---

## 3. FDST Dataset — `datasets/fdst.py`

### 3.1 `__init__`: 掃描 images 目錄

```python
images_dir = f"{root}/{mode}_data/images"
# 檢測子目錄 (影片ID) 或扁平結構
self.frames.append((vid, frame_name))   # [(vid, frm), ...]

# Protocol 決定 labeled 集
idset = set(protocol_lines)
if (not self.training) or (idset is None) or (key in idset):
    self.label.append((vid, frm))
self.unlabel.append((vid, frm))          # 所有幀也在 unlabel

label_to_idx / unlabel_to_idx dict   # O(1) lookup
dot_dirs = ('new-anno', 'annotations')   # 標註目錄
```

### 3.2 `__getitem__` (訓練)

```python
lid = random.choice(self.label)               # 隨機 labeled 幀
limg_seq, lseqs = readLabelSequenceFromTuple(lid)   # (T,3,256), [dot, ...]
uid = self.unlabel[index % len(self.unlabel)]       # 按 index
uimg_seq, umask_seq = readUnlabelSequenceFromTuple(uid)  # (T,3,256), (T,1,256)
uflows = _load_flow_for_sequence(unlabel, index)    # (T-1,2,64,64) or None
return 7-tuple
```

### 3.3 序列讀取

```python
def readLabelSequenceFromTuple(self, tpl):
    idx = self.label_to_idx[tpl]
    idxs = [start, start+1*stride, ...]    # 取 seq_len 個連續幀
    for ii in idxs:
        img = _load_image(vid, frm)         # PIL.Open + RGB
        img_t = self.norm_func.im2tensor(img) # ToTensor + Normalize
        dot = _load_annotation(vid, frm)     # GT_{vid}_{frm}.npy → (N,2)
        img_t, dot = process_lable(img_t, dot)  # 增強管線
        imgs.append(img_t.squeeze(0))         # (3,256,256)
        dotseqs.append(dot)                   # [tensor_patch0] ← list of 1
    imgs_seq = torch.stack(imgs)              # (T, 3, 256, 256)
    return imgs_seq, dotseqs                  # dotseqs = [[tensor_f0], ...]
```

### 3.4 光流載入

```python
fn = os.path.join(flow_root, f"{vid}_{frm}_flow{ext}")  # 固定命名
arr = np.load(fn, mmap_mode='r')             # (2, H/4, W/4) float32
flows.append(torch.from_numpy(arr).float())
return torch.stack(flows, dim=0)             # (T-1, 2, H/4, W/4)
```

### 3.5 collate_fn

```python
limg_batch = torch.stack(limg_seqs)    # (B, T, 3, 256, 256)
uimg_batch = torch.stack(uimg_seqs)    # (B, T, 3, 256, 256)
umask_batch = torch.stack(umask_seqs)  # (B, T, 1, 256, 256)
uflows_batch = torch.stack(flows) or None  # (B, T-1, 2, 64, 64)
return 7-tuple
```

---

## 4. 資料增強 — `datasets/utils.py`

### 4.1 `process_lable` (labeled 幀)

```python
if self.train:
    images, dotseqs = self.crop_and_resize(image, dotseq)
else:
    images, dotseqs = image.unsqueeze(0), [dotseq]
    # resize to (256,256) + 點座標縮放 (for eval)

F.pad(to_32_alignment)           # pad 至 32 倍數
random_horizontal_flip()          # 50% 翻轉

for seq in dotseqs:
    dotseqs[i] = seq[:, [1, 0]]   # ⚠️ swap (x,y) → (y,x)

return images, dotseqs            # images: (1,3,256,256), dotseqs: [tensor(N,2)]
```

### 4.2 `crop_and_resize` (CPU 主力)

```python
def crop_and_resize(self, image, dotseq=None, num_patches=1):
    scale = random() * 0.6 + 0.7            # 隨機縮放 0.7~1.3
    crop_h, crop_w = 256/scale, 256/scale
    if crop_h > imh: F.pad(image, ...)       # 超出則 pad
    start_h = random.randint(0, imh - crop_h)
    start_w = random.randint(0, imw - crop_w)
    crop_img = image[:, start_h:end_h, start_w:end_w]
    crop_imgs = F.interpolate(crop_imgs, (256,256), mode='bilinear')  # CPU

    if dotseq is not None:
        idx = (w>=start_w) & (w<=end_w) & (h>=start_h) & (h<=end_h)
        selected_dot = dotseq[idx]
        selected_dot[:, 0] = (x - start_w) * (256/crop_w)  # 座標縮放
        selected_dot[:, 1] = (y - start_h) * (256/crop_h)
    return crop_imgs, crop_dots              # (1,3,256,256), [tensor(N',2)]
```

---

## 5. 模型 — `models/vgg16bn.py`

```python
class VGG16_BN(nn.Module):
    def __init__(self):
        self.encoders = ModuleList([Sequential(features[0:33]),
                                    Sequential(features[33:43])])
        self.fuse_layer = UpSample_P2P([512,512], ouc=256)
        self.decoders = SimpleDecoder(256, 256, up_scale=2, out_channel=1)
        self.to_temporal = Conv2d(1, 1, 1)
        self.temporal = TemporalUnit(mode='convgru', in_ch=1, hid_ch=32)
        self.hidden2logit = Conv2d(32, 1, 1)
        self.down = 4

    def forward(self, image, prev_h=None):
        fea2 = self.encoding(image)         # VGG16BN extract
        denmap = self.decoding(fea2)         # → (B,1,H/4,W/4)
        x_t = self.to_temporal(denmap)
        next_h = self.temporal(x_t, prev_h)  # ConvGRU
        pred_logits = self.hidden2logit(next_h)  # (B,1,H/4,W/4)
        return pred_logits, next_h
```

### ConvGRU Cell (`models/utils.py`)

```python
class ConvGRUCell(nn.Module):
    def forward(self, x, h):
        if h is None:
            h = zeros(batch, hid_ch, H, W)   # 自初始化
        z, r = sigmoid(conv_zr(cat)).chunk(2)
        n = tanh(conv_n([x, r*h]))
        h_next = (1-z) * n + z * h
```

---

## 6. P2R Loss (向量化 + Chunked) — `losses/p2rloss.py`

### 核心演算法

```python
CHUNK_SIZE = 64

def _process_chunk(self, A_chunk, B_coord_chunk, point_valid_chunk, A_coord, HW, down):
    # 1. 最近鄰距離 (GPU)
    dist2 = cdist(B_flat, B_flat).pow(2)   # (cnb, max_N, max_N)
    dist2.fill_diagonal_(inf)
    nearest_dist = dist2.min(dim=-1)

    # 2. 距離矩陣 C: (cnb, 1, HW, max_N)
    C = |A_coord - B_coord_chunk|

    with torch.no_grad():
        # 3. Round 1: 每個 pixel → 最近 GT 點
        minC, mcidx = C.min(dim=-1)         # (cnb, 1, HW, 1)
        M = scatter_(mcidx, 1.0) * (C < max_radis)  # 配對遮罩

        # 4. 歸一化
        maxC = clip((minC * M).amax(), min_radis, max_radis)
        C = C / maxC
        C = C * cost_point - A * cost_class

        # 5. 過濾無效點 (vid)
        vid = (M.sum(dim=2) > 0) & point_valid
        C = C * vid.unsqueeze(-2)

        # 6. Round 2: 每個 GT 點 → 最佳 pixel
        C2 = M * C + (1-M) * (C.max() + 1)
        minC2, mcidx2 = C2.min(dim=2)        # 每點選最佳 pixel
        T = scatter_(mcidx2, 1.0).sum(-1)    # 目標遮罩
        T = (T > 0.5).float()
        W = T + 1.0                           # 權重: pos=2, neg=1

    return T, W
```

### forward (batch 拆 chunk)

```python
def forward(self, dens, seqs, down, ...):
    A = dens.view(bs, HW)                     # logits, shape (bs, HW)
    T_full = zeros(bs, HW)
    W_full = ones(bs, HW) * 0.5               # 空樣本預設權重

    for chunk_start in range(0, nb, CHUNK_SIZE):  # 拆 64 一組
        T_chunk, W_chunk = _process_chunk(...)
        T_full[chunk_idx] = T_chunk
        W_full[chunk_idx] = W_chunk

    # ⚠️ BCE loss: mean over ALL pixels
    loss = BCEWithLogitsLoss(A, T_full, weight=W_full, reduction='mean')
    return loss
```

### 關鍵公式

```
loss = BCEWithLogits(A, T, weight=W)   # A: logits, T: 0/1 target, W: 1 or 2

# T=0 的 pixel: loss 最小 → A 趨近 -∞
# T=1 的 pixel: loss 最小 → A 趨近 +∞
# weight=2 的 T=1 pixel: 正樣本權重 2 倍
```

---

## 7. 訓練循環 — `train/train_loop.py`

### Stage 1 (sup)

```python
sup_loss = p2r(lpred, ldots)    # 有標註第一幀 → P2R vs GT dots
loss = sup_loss
backward + optimizer.step()
# 無 teacher, 無時序, 無 EMA
```

### Stage 2 (semi)

```python
sup_loss = p2r(lpred, ldots)     # 有標註第一幀 (同 Stage 1)

# 初始化 hidden state (ConvGRU)
student_h = student.init_hidden(B, device, (H//4, W//4))
teacher_h = teacher.init_hidden(B, device, (H//4, W//4))

# Teacher t=0 預測
teacher_pred0 = teacher(uimg_batch[:,0], teacher_h)

for t in range(1, T):            # T=2: 僅 1 次迭代
    optimizer.zero_grad()

    # Student t 預測
    student_pred_t = student(uimg_batch[:,t], student_h)

    # Teacher pseudo-points from t-1
    points_prev = generate_points_from_density(teacher_pred_prev, threshold=0.3)
    if flows: warped = warp_points_with_flow(points_prev, flow_batch[:,t-1])
    loss_p2r = p2r(student_pred_t, warped)

    loss = loss_p2r + sup_loss / (T-1)    # T=2 → sup_loss 的比重
    backward + scaler.step()
    student_h.detach()
    update_ema(student, teacher, 0.999)   # EMA
    teacher_pred_t = teacher(uimg_batch[:,t], teacher_h)  # teacher t 預測 (下一輪用)
```

---

## 8. 主入口 — `main.py`

### 8.1 兩階段控制

```python
STAGE_1 = 25     # Epoch 0-24: supervised only
STAGE_2 = 50     # Epoch 25-49: semi-supervised

for epoch in range(START_EPOCH, EPOCHS):
    stage = 'sup' if epoch < 25 else 'semi'

    if stage == 'semi' and epoch == 25 and not resumed:
        teacher.load_state_dict(student.state_dict())  # Teacher ← student
        torch.cuda.empty_cache()

    temporal_train_one_epoch(..., stage=stage)

    # 每 5 epoch validate
    if epoch % 5 == 0 or epoch == EPOCHS-1:
        mae, mse, loss = validate(data_loader_val, student, test_criterion)
        plot_curve(...)
```

### 8.2 Validate (計數估計)

```python
def validate(config, data_loader, model, criterion):
    for batch in data_loader:
        images = batch[0]                         # (B, T, 3, H, W) or (B, 3, H, W)
        if images.dim() == 5:
            images = images[:, 0]                  # 只取第一幀 ⚠️
        dotseq = [s[0][0].cuda(...) for s in batch[1]]  # 第一幀 dots

        output, _ = model(images, prev_h=None)    # 無 temporal context ⚠️
        loss = criterion(output, dotseq, down).item()

        outnum = (output > 0).sum(dim=(1,2,3))    # ⚠️ 正 logit 計數 (非 density sum)
        mae = |outnum - cnt|.mean()
        mse = (|outnum - cnt|.pow(2)).mean()

    return mae, mse, loss
```

---

## 9. Loss 不收斂原因分析

### 🚨 原因 1: Validate 計數方式錯誤（但這不影響訓練，只影響曲線）

```python
outnum = (output > 0).sum()    # 正 logit 像素計數
```

標準人群計數: `outnum = output.sum() / down²`。當前計數無物理意義，MAE/MSE 曲線無法反映真實收斂。

### 🚨 原因 2: Learning Rate 從不衰減

```python
lr_scheduler = StepLR(optimizer, step_size=3500, gamma=0.9)  # DECAY_EPOCHS=3500
```

`EPOCHS=50` (或 100) << `DECAY_EPOCHS=3500` → LR 始終為 `5e-5` 不變。模型可能收斂到局部極小後無法越過。

### 🚨 原因 3: Stage 2 的 pseudo-point 品質未驗證

```python
points_prev_list = generate_points_from_density(teacher_pred_prev, threshold=0.3)
```

Teacher 產生的 pseudo-point 數量/品質直接決定 `loss_p2r` 的訓練信號。若 teacher 輸出 flat（無明顯 peak），`generate_points_from_density` 可能產生：
- **過多點** (uniform output → 全為 local maxima) → loss_p2r 變成隨機監督
- **過少點** (output too flat) → loss_p2r ≈ BCE(zeros, weight=0.5) → 把所有 logit 推向 -∞

**從 loss 曲線看**: loss 穩定在 ~0.33 不變，可能是 teacher & student 達到一個平衡態 →
- 所有 pixel logit ≈ -1 (mean BCE ≈ 0.31)
- 少數 positive pixel logit ≈ +2 (mean BCE ≈ 2.13)
- 整體 mean ≈ (0.31×4095 + 2.13×1) / 4096 ≈ 0.31
- 這個平衡態無法進一步改善，因為正樣本 (T=1) 佔比極低 (~0.1%)

### 🚨 原因 4: BCE 正負樣本極度不平衡

在 64×64 density map 中:
- T=1 pixels: 約等於 GT 點數 (N=50~500) / 4096 = 1%~12%
- T=0 pixels: 88%~99%

BCE with `reduction='mean'` 使 loss 由**絕大多數的負樣本主導**。`weight=T+1` 只給正樣本 weight=2（vs 負樣本 weight=1），差異太小。

### 🚨 原因 5: Stage 2 的 sup_loss 與 loss_p2r 尺度可能不對齊

```python
loss = loss_p2r + sup_loss / (T - 1)    # T=2 → sup_loss / 1 = sup_loss
```

`sup_loss` 來自 labeled 幀 GT 真實點。
`loss_p2r` 來自 unlabeled 幀 teacher pseudo-points (可能稀疏或偏移)。

若 `loss_p2r` 持續遠小於 `sup_loss`（因 pseudo-points 極少），則總 loss ≒ `sup_loss`，Stage 2 等同於 Stage 1。

### 🚨 原因 6: `teacher_pred_prev` 在時間步間傳遞，但 student 的梯度不影響 teacher

```python
# 時間 t:
teacher_pred_prev → generate_points → pseudo-points → loss_p2r
# ↑ teacher 由 EMA 更新，但 teacher_pred_prev 來自上一個時間步的教師預測
```

梯度只通過 `student` 傳播。Teacher 的預測 `teacher_pred_prev` 由 EMA 緩慢跟隨。若 teacher 初始化不佳，首幾輪的 pseudo-points 會嚴重偏差，損害訓練。

### 🚨 原因 7: 無 lr scheduler 實際作用

```python
lr_scheduler = StepLR(optimizer, step_size=3500, gamma=0.9)
```

即使在 `main_worker` 中建立了 `lr_scheduler`，但在訓練循環中沒有 `lr_scheduler.step()`。LR 完全固定。

### 總結

| 問題 | 嚴重性 | 影響 |
|------|--------|------|
| LR 固定不衰減 | 🔴 高 | 無法 fine-tune |
| BCE 正負樣本不平衡 | 🔴 高 | loss 由負樣本主導 |
| pseudo-point 品質未驗證 | 🟡 中 | Stage 2 可能無效 |
| validate 計數方式錯誤 | 🟡 中 | 曲線不反映真實收斂 |
| Stage 2 兩 loss 尺度未對齊 | 🟡 中 | 退化成 Stage 1 |
| teacher 初始化無 warmup | 🟢 低 | Stage 2 初期不穩定 |
