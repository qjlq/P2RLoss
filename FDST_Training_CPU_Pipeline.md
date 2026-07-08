# FDST 時序人群計數訓練 — CPU 處理管線全覽

## 文件結構

```
P2RLoss/
├── config.py                配置定義（NUM_WORKERS, SEQ_LEN, FLOW_ROOT 等）
├── main.py                  訓練入口（兩階段訓練, 每5 epoch validate, 進度條）
├── datasets/
│   ├── __init__.py          DataLoader 建立（file_system策略, persistent_workers, prefetch）
│   ├── fdst.py              FDST 資料集（序列讀取, 光流加載, O(1) 字典查找, mmap）
│   └── utils.py             CPU 資料增強管線（NormalSample, crop_and_resize, nearest）
├── models/
│   ├── vgg16bn.py           VGG16BN + ConvGRU（GPU）
│   └── utils.py             ConvGRUCell, UpSample_P2P, SimpleDecoder（GPU）
├── losses/
│   └── p2rloss.py           P2RLoss 向量化（GPU, 無 per-sample loop）
├── train/
│   └── train_loop.py        訓練循環（tqdm, EMA, BPTT, flow warp, 進度條）
└── preprocess/
    └── preprocess_fdst.py   離線預處理（VIA JSON → GT_*.npy, Farneback 光流, 重採樣）
```

---

## 一、CPU 資料管線總覽（每個 __getitem__ 呼叫）

```
                                 DataLoader Worker (CPU)
  ┌─────────────────────────────────────────────────────────────────────┐
  │                                                                     │
  │  __getitem__(index)                                                  │
  │    │                                                                 │
  │    ├─ random.choice(self.label)          ← O(1) 隨機選 labeled 幀   │
  │    │                                                                 │
  │    ├─ readLabelSequenceFromTuple(lid)                                │
  │    │    ├─ label_to_idx[tpl]              ← O(1) dict 查找          │
  │    │    ├─ _get_sequence_indices()        ← 生成 seq_len 個索引     │
  │    │    └─ for 每幀:                                                 │
  │    │         ├─ _load_image()              ⬅ DISK I/O (PIL.Open)     │
  │    │         ├─ im2tensor()                ⬅ CPU: ToTensor+Norm     │
  │    │         ├─ _load_annotation()         ⬅ DISK I/O (mmap .npy)   │
  │    │         └─ process_lable()           ⬅ CPU 增強管線 (見§三)    │
  │    │                                                                 │
  │    ├─ readUnlabelSequenceFromTuple(uid)                              │
  │    │    ├─ unlabel_to_idx[tpl]             ← O(1) dict 查找          │
  │    │    ├─ _get_sequence_indices()                                   │
  │    │    └─ for 每幀:                                                 │
  │    │         ├─ _load_image()              ⬅ DISK I/O                │
  │    │         ├─ im2tensor()                ⬅ CPU: ToTensor+Norm     │
  │    │         ├─ process_unlabel()         ⬅ CPU 增強管線 (見§四)    │
  │    │         └─ random_mask()              ⬅ CPU: 隨機矩形遮罩      │
  │    │                                                                 │
  │    └─ _load_flow_for_sequence()                                      │
  │         └─ for 每幀對:                                               │
  │              ├─ os.path.exists()            ⬅ DISK I/O (路徑檢查)    │
  │              └─ np.load(mmap_mode='r')      ⬅ DISK I/O (mmap .npy)  │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    collate_fn (CPU, main process)
                      torch.stack() × 4
                      flow padding
                              │
                              ▼
                    .to(device) → GPU
```

---

## 二、DataLoader 配置 — `datasets/__init__.py`（CPU Worker 管理）

```python
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')  # 避免 FD 耗盡

DataLoader(
    dataset,
    batch_size=16,              # 每個 batch 16 個 samples
    num_workers=4,              # 4 個 CPU worker 並行載入
    shuffle=True,               # 訓練時 shuffle
    collate_fn=FDST.collate_fn, # 自定義批次組裝
    persistent_workers=True,    # worker 跨 epoch 持續存活（不重建）
    prefetch_factor=4,          # 每個 worker 預先 load 4 個 batch
)
```

### CPU Worker 產能估算

```
參數: batch_size=16, num_workers=4, seq_len=2
每個 worker 每次處理: 16÷4 = 4 個 samples
每個 sample: 2(L) + 2(U) = 4 幀 + 1 個 flow
每幀 CPU 時間 ≈ 50-200ms（含 I/O + 增強）
每 batch CPU 時間 ≈ 4 × 4 × 100ms = 1.6s（4 workers 平行）
```

---

## 三、有標註幀 CPU 增強管線 — `datasets/utils.py:process_lable`

```
process_lable(image_tensor, dot_tensor)
  │
  ├─ crop_and_resize()                    ← 隨機縮放 + 裁剪
  │    ├─ random() × scale_factor         ← 隨機縮放倍率 0.7~1.3
  │    ├─ F.pad()                          ← 若裁剪範圍超出原圖
  │    ├─ random_crop()                    ← 隨機位置裁剪
  │    ├─ F.interpolate((256,256))         ⬅ CPU: 雙線性插值 resize
  │    └─ 點座標轉換: select + scale      ⬅ CPU: 篩選框內點 + 縮放
  │
  ├─ F.pad(to_32_alignment)               ⬅ CPU: pad 至 32 倍數
  │
  ├─ random_horizontal_flip()             ⬅ CPU: 50% 機率水平翻轉
  │
  └─ nearest()                            ⬅ CPU: 每點最近鄰距離
       ├─ torch.cdist(seq, seq)            ⬅ CPU: O(N²) 距離矩陣
       ├─ fill_diagonal_(inf)              ← 排除自身
       └─ min(dim=1)                       ← 取最近鄰
```

### `crop_and_resize` 詳細流程 (`datasets/utils.py:84-132`)

```python
def crop_and_resize(self, image, dotseq=None, num_patches=1):
    # image: (3, H, W) CPU tensor, dotseq: (N, 2) CPU tensor

    scale = random.random() * 0.6 + 0.7       # 隨機縮放 0.7~1.3
    crop_h = int(256 / scale)
    crop_w = int(256 / scale)

    if crop_h > imh or crop_w > imw:
        image = F.pad(image, ...)              # ⬅ CPU: 超出則 pad

    start_h = random.randint(0, imh - crop_h)  # 隨機起始點
    start_w = random.randint(0, imw - crop_w)

    crop_img = image[:, start_h:end_h, start_w:end_w]  # 裁剪
    crop_imgs = F.interpolate(                         # ⬅ CPU: bilinear
        crop_imgs, (256, 256), mode='bilinear')

    if dotseq is not None:
        idx = (dotseq[:,0] >= start_w) & ...  # 篩選框內點
        selected_dot = dotseq[idx]
        selected_dot[:, 0] = (x - start_w) * (256 / crop_w)  # 座標縮放
        selected_dot[:, 1] = (y - start_h) * (256 / crop_h)

    return (1, 3, 256, 256), [tensor(N', 2)]
```

### `nearest` 詳細流程 (`datasets/utils.py:134-141`)

```python
def nearest(self, seq):
    # seq: (N, 2) CPU tensor  in (x, y) 座標
    if seqlen <= 1:
        return torch.full((1, 1), 32.0)

    dist2 = torch.cdist(seq, seq).pow(2)       # ⬅ (N,N) 距離矩陣
    dist2.fill_diagonal_(float('inf'))          # 排除自身
    m = dist2.min(dim=1).values                 # ⬅ 最近鄰距離
    return m.view(-1, 1)                        # (N, 1)

# 最終 dot = torch.cat((seq[:, [1,0]], u), dim=1)
# seq[:, [1,0]]: 交換 x,y → (y, x)
# u: 最近鄰距離
# dot: (N, 3) = (y, x, nn_dist)
```

---

## 四、無標註幀 CPU 增強管線 — `datasets/utils.py:process_unlabel`

```
process_unlabel(image_tensor)
  │
  ├─ crop_and_resize(image)                  ← 同 labeled 的隨機裁剪
  │    └─ 無 dot → 只回傳 crop_imgs
  │
  ├─ F.pad(to_32_alignment)                 ⬅ CPU: pad 至 32 倍數
  │
  └─ random_horizontal_flip()               ⬅ CPU: 50% 機率翻轉
```

註：原始的 `strong_aug`（ColorJitter + GaussianBlur）已移除（變數未使用）。

### 隨機遮罩 (`datasets/fdst.py:258-265`)

```python
def random_mask(self, uimgs):
    # uimgs: (1, 3, 256, 256) CPU tensor
    mask = torch.ones((1, 1, 256, 256))

    cut_w = int(256 * random(1/8, 1/4))    # 32~64 像素
    cut_h = int(256 * random(1/8, 1/4))
    top = random.randint(0, 256 - cut_h)
    left = random.randint(0, 256 - cut_w)

    mask[:, :, top:top+cut_h, left:left+cut_w] = 0   # 矩形遮罩
    return mask                              # (1, 1, 256, 256)
```

---

## 五、磁碟 I/O 明細 — `datasets/fdst.py`

### 5.1 圖片載入

```python
def _load_image(self, vid, frm):
    # 嘗試路徑: {img_dir}/{vid}/{frm}.jpg → .png
    return Image.open(imgpath).convert('RGB')   # ⬅ PIL 解碼 JPEG/PNG
```

### 5.2 標註載入（mmap）

```python
def _load_annotation(self, vid, frm):
    candidates = [                            # 嘗試 4 種路徑
        "new-anno/GT_{vid}_{frm}.npy",
        "new-anno/GT_{frm}.npy",
        "annotations/{vid}/{frm}.npy",
        "annotations/{frm}.npy",
    ]
    return torch.from_numpy(
        np.load(p, mmap_mode='r').copy()       # ⬅ mmap 讀取 .npy
    )[:, :2]                                   # (N, 2) = (x, y)
```

### 5.3 光流載入（mmap）

```python
def _load_flow_for_sequence(self, list_ref, start_idx):
    if not self.flow_root:
        return None

    for i in range(seq_len - 1):   # SEQ_LEN=2 → 1 個 flow
        f0 = list_ref[idxs[i]]
        cand1 = f"{flow_root}/{vid}_{frm}_flow{ext}"   # 子目錄命名
        cand2 = f"{flow_root}/{frm}_flow{ext}"          # 扁平命名

        if os.path.exists(cand1): fn = cand1             # ⬅ stat 系統呼叫
        elif os.path.exists(cand2): fn = cand2

        arr = np.load(fn, mmap_mode='r')                 # ⬅ mmap 讀取
        flows.append(torch.from_numpy(arr).float())      # → (2, H, W) float32

    return torch.stack(flows, dim=0)   # (T-1, 2, H/4, W/4)
```

### 5.4 O(1) 字典查找（避免 O(N) list.index）

```python
# __init__ 中建立（一次建構）
self.label_to_idx = {tpl: i for i, tpl in enumerate(self.label)}
self.unlabel_to_idx = {tpl: i for i, tpl in enumerate(self.unlabel)}

# 使用時（O(1) hash lookup，取代 O(N) .index()）
idx = self.label_to_idx[tpl]        # 原: self.label.index(tpl)
idx = self.unlabel_to_idx[tpl]      # 原: self.unlabel.index(tpl)
```

### 5.5 collate_fn（CPU 批次組裝）

```python
@staticmethod
def collate_fn(samples):              # ⬅ 主 process 執行
    # 拆包 7-tuple
    limg_batch = torch.stack(limg_seqs)     # (B, T, 3, 256, 256)
    uimg_batch = torch.stack(uimg_seqs)     # (B, T, 3, 256, 256)
    umask_batch = torch.stack(umask_seqs)   # (B, T, 1, 256, 256)

    # 光流處理: None → zeros padding
    uflows_batch = torch.stack(stacked)     # (B, T-1, 2, 64, 64) or None

    return limg_batch, lseqs, lids, uimg_batch, umask_batch, uids, uflows_batch
```

---

## 六、訓練循環 — `train/train_loop.py`（CPU 與 GPU 交錯）

### 6.1 每個 batch 的 CPU/GPU 分工

```
for batch in dataloader:            # ⬅ CPU: DataLoader 阻塞等待
    ├─ .to(device)                   ⬅ GPU: CPU→GPU 傳輸
    ├─ student.forward               ⬅ GPU
    ├─ teacher.forward               ⬅ GPU
    ├─ generate_points_from_density  ⬅ GPU (sigmoid + maxpool + nonzero)
    ├─ warp_points_with_flow         ⬅ GPU (grid_sample)
    ├─ P2R loss                      ⬅ GPU (向量化張量運算)
    ├─ scaler.backward/step          ⬅ GPU
    ├─ student_h.detach()            ⬅ CPU (張量元數據操作)
    ├─ update_ema()                  ⬅ CPU (參數逐元素 mul_/add_)
    └─ teacher.forward               ⬅ GPU
```

### 6.2 EMA 更新（CPU 參數操作）

```python
def update_ema(student, teacher, alpha=0.999):
    # ⬅ CPU: 逐參數迭代 + 就地張量運算
    for sp, tp in zip(student.parameters(), teacher.parameters()):
        tp.data.mul_(alpha).add_(sp.data, alpha=1.0 - alpha)
    # VGG16BN ~20M 參數, 每次呼叫 ~2-5ms
```

### 6.3 進度條（CPU tqdm）

```python
pbar = tqdm(dataloader, desc=f'Epoch [{epoch}]', leave=False)
# tqdm 包裝 DataLoader, 每次迭代.next() 即從 worker 索取下一 batch
# 若 worker 尚未準備好 → 進度條阻塞（GPU idle）
```

### 6.4 Pseudo-points 生成（GPU）

```python
def generate_points_from_density(density_logits, threshold=0.3):
    # 全 GPU 操作
    prob = torch.sigmoid(density_logits)
    pool = F.max_pool2d(prob, kernel_size=3, padding=1)
    peaks = (prob == pool) & (prob > threshold)

    # Python loop over batch（微量 CPU 開銷）
    for b in range(B):
        ys, xs = torch.nonzero(peaks[b, 0], as_tuple=True)
        coords = torch.stack([xs.float(), ys.float()], dim=1)
        points_list.append(coords)
    return points_list
```

### 6.5 Flow Warp（GPU）

```python
def warp_points_with_flow(points_list, flow_batch):
    # ⬅ GPU: grid_sample 雙線性採樣
    sampled = F.grid_sample(flow, grid, align_corners=True)

    # ⬅ CPU: Python loop over batch（少量）
    for b in range(B):
        warped_pts = points_list[b] + flow_vecs
    return warped_list
```

---

## 七、完整資料流與張量形狀

```
FDST 磁碟結構:
  {root}/train_data/images/{video_id}/{frame}.jpg         ← 原始圖片
  {root}/train_data/new-anno/GT_{video_id}_{frame}.npy     ← (N,2) 點標註
  {root}/precomputed_flow/{video_id}_{frame}_flow.npy      ← (2, H/4, W/4) 光流

__getitem__ 輸出:
  limg_seq:   (T, 3, 256, 256)     labeled 影像序列
  lseqs:      list[T] of tensor(N, 3)  點標註 (y, x, nn_dist)
  uimg_seq:   (T, 3, 256, 256)     unlabeled 影像序列
  umask_seq:  (T, 1, 256, 256)     隨機遮罩
  uflows:     (T-1, 2, 64, 64)     光流 (down=4) 或 None

collate_fn 輸出:
  limg_batch:   (B, T, 3, 256, 256)
  uimg_batch:   (B, T, 3, 256, 256)
  umask_batch:  (B, T, 1, 256, 256)
  uflows_batch: (B, T-1, 2, 64, 64) 或 None

train loop (stage='semi', T=2):
  Teacher t=0 pred:   (B, 1, 64, 64)
  Student t=1 pred:   (B, 1, 64, 64)
  Teacher t=0 pts:    list[B] of (N_i, 2)
  Warped pts:         list[B] of (N_i, 2)
  sup_loss + loss_p2r → loss scalar

validate:
  images[:, 0]       (B, 3, 256, 256)     # 取序列第一幀
  output:            (B, 1, 64, 64)
  outnum:            (output > 0).sum()    # 計數估計
  mae/mse:           |outnum - cnt| mean
```

---

## 八、性能瓶頸總結

| 階段 | 裝置 | 耗時佔比 | 瓶頸說明 |
|------|------|---------|---------|
| **圖像讀取** | CPU+DISK | ~20% | PIL JPEG 解碼, 依賴磁碟速度 |
| **crop_and_resize** | CPU | ~15% | `F.interpolate` bilinear + 點座標轉換 |
| **最近鄰計算** | CPU | ~10% | `torch.cdist` O(N²), 密集幀 N=300-500 時顯著 |
| **標註/光流載入** | CPU+DISK | ~10% | `np.load` + `os.path.exists`, mmap 後為記憶體速度 |
| **Worker IPC** | CPU | ~5% | 多進程 queue 通訊 + collate 組裝 |
| **GPU 前向/反向** | GPU | ~40% | VGG16BN + ConvGRU 4 次 forward + 1 次 backward |
| **EMA 更新** | CPU | <1% | 20M 參數 mul_/add_ |
| **DataLoader 等待** | CPU | 不定 | **最大瓶頸**：Worker 不足時 GPU idle |

### 瓶頸排除順序

```
1. CPU Worker 不足     → 增加 num_workers + prefetch_factor
2. DISK I/O 競爭       → 使用 RAMDisk (`/dev/shm/`) 或 SSD + mmap
3. 增強管線過重         → crop_and_resize 已為必需, nearest() 已用 cdist 優化
4. O(N) list.index()   → 已用 O(1) dict 替換
5. 無用計算             → strong_aug 已移除
```

### 建議的執行命令

```bash
cp -r /path/to/FDST /dev/shm/                          # RAMDisk 繞過硬碟
python main.py --data-path /dev/shm/FDST \
    --label 5 --protocol /path/to/protocol.txt \
    --batch-size 16 --tag fdst-fast \
    --opts DATA.DATASET fdst \
           DATA.SEQ_LEN 2 DATA.SEQ_STRIDE 1 \
           DATA.FLOW_ROOT /dev/shm/FDST/precomputed_flow \
           DATA.NUM_WORKERS 4
```
