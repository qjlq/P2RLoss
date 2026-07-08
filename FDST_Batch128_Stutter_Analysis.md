# Batch=128 每 6 batch 卡頓 — 代碼分析

## 相關代碼文件總覽

```
訓練流程（每個 batch）:
  DataLoader (CPU Workers)
    ├─ __getitem__ × 128 (8 workers × 16 輪)
    │    ├─ readLabelSequenceFromTuple  (labeled 序列 + 增強)
    │    ├─ readUnlabelSequenceFromTuple (unlabeled 序列 + 增強)
    │    └─ _load_flow_for_sequence     (光流 mmap 讀取)
    │
    ├─ collate_fn (CPU 主程序)          ← torch.stack × 4 + flow padding
    │
    └─ .to(device) × 4                  ← CPU→GPU 傳輸

  GPU 計算
    ├─ student.forward (labeled)         ← VGG16BN
    ├─ teacher.forward (t=0)             ← VGG16BN
    ├─ student.forward (t=1)             ← VGG16BN
    ├─ generate_points_from_density      ← GPU (sigmoid + maxpool + nonzero loop)
    ├─ warp_points_with_flow             ← GPU (grid_sample)
    ├─ p2r (sup_loss)                    ← GPU (向量化, chunk=16)
    ├─ p2r (loss_p2r)                    ← GPU (向量化, chunk=16)
    ├─ scaler.scale + backward           ← GPU
    ├─ scaler.step + update              ← GPU
    ├─ update_ema                        ← CPU (20M 參數 mul_/add_)
    └─ teacher.forward (t=1)             ← GPU
```

---

## 1. DataLoader 建立 — `datasets/__init__.py:31-40`

```python
DataLoader(
    dataset,
    batch_size=128,                # 每批 128 samples
    num_workers=8,                 # 8 個 CPU worker
    shuffle=True,
    collate_fn=FDST.collate_fn,
    persistent_workers=True,        # worker 跨 epoch 持續
    prefetch_factor=4,             # ⚠️ 每個 worker 預取 4 個 indices
)
```

### 關鍵計算

```
初始 index queue = num_workers × prefetch_factor = 8 × 4 = 32
每 batch 需求   = batch_size = 128 samples
每 worker 貢獻  = 128 / 8 = 16 samples
初始覆蓋率      = 32 / 128 = 25% ← 僅覆蓋四分之一 batch
```

---

## 2. 每個 __getitem__ 的 CPU 工作 — `datasets/fdst.py:118-131`

```python
def __getitem__(self, index):
    lid = random.choice(self.label)                  # O(1) 隨機選
    limg_seq, lseqs = self.readLabelSequenceFromTuple(lid)  # 2 幀 labeled
    uid = self.unlabel[index % len(self.unlabel)]     # 順序 unlabeled
    uimg_seq, umask_seq = self.readUnlabelSequenceFromTuple(uid) # 2 幀 unlabeled
    uflows = self._load_flow_for_sequence(...)        # 1 個 flow (mmap)
    return (7 items)
```

### 每 sample 載入量

```
labeled 序列: 2 幀 × (PIL Open + ToTensor + crop_and_resize + pad + flip)
unlabeled 序列: 2 幀 × (PIL Open + ToTensor + crop_and_resize + pad + flip + random_mask)
flow: 1 個 × mmap 讀取 .npy (2, 64, 64) float32 → 32KB
```

### 128 batch 總載入量

```
圖檔讀取: 128 × 4 = 512 張 JPEG → PIL 解碼（最重的 I/O）
flow 讀取: 128 × 1 = 128 個 .npy → mmap（虛擬記憶體，頁錯誤延遲）
```

---

## 3. 序列讀取 — `datasets/fdst.py:143-179`

```python
def readLabelSequenceFromTuple(self, tpl):
    idx = self.label_to_idx[tpl]                     # O(1) dict lookup
    idxs = self._get_sequence_indices(self.label, idx)  # [idx, idx+1, ...]
    for ii in idxs:                                   # 2 次 (SEQ_LEN=2)
        img = self._load_image(vid, frm)              # PIL.Open + JPEG decode ⬅ I/O
        img_t = self.norm_func.im2tensor(img)          # ToTensor + Normalize
        dot = self._load_annotation(vid, frm)          # mmap .npy ⬅ I/O
        img_t, dot = self.norm_func.process_lable(img_t, dot)  # 增強管線
        imgs.append(img_t.squeeze(0))
    imgs_seq = torch.stack(imgs, dim=0)

def readUnlabelSequenceFromTuple(self, tpl):
    for ii in idxs:
        img = self._load_image(vid, frm)              # PIL.Open + JPEG decode ⬅ I/O
        wa_img = self.norm_func.im2tensor(img)
        img_proc = self.norm_func.process_unlabel(wa_img)  # 增強管線
        masks.append(self.random_mask(img_proc).squeeze(0))
    imgs_seq = torch.stack(imgs, dim=0)
```

---

## 4. 資料增強 — `datasets/utils.py`

### crop_and_resize (CPU 重操作)

```python
def crop_and_resize(self, image, dotseq=None, num_patches=1):
    scale = random() * 0.6 + 0.7               # 隨機縮放
    crop_h = int(256 / scale)
    crop_w = int(256 / scale)
    # 隨機裁剪
    start_h, start_w = random.randint(...)
    crop_img = image[:, start_h:end_h, start_w:end_w]
    crop_imgs = F.interpolate(                  # ⬅ CPU bilinear resize
        crop_imgs, (256, 256), mode='bilinear')
    return crop_imgs, crop_dots
```

---

## 5. collate_fn（CPU 主程序）— `datasets/fdst.py:271-311`

```python
@staticmethod
def collate_fn(samples):
    limg_batch = torch.stack(limg_seqs)   # (128, 2, 3, 256, 256) ≈ 192 MB
    uimg_batch = torch.stack(uimg_seqs)   # (128, 2, 3, 256, 256) ≈ 192 MB
    umask_batch = torch.stack(umask_seqs)  # (128, 2, 1, 256, 256) ≈ 64 MB
    # flow 處理
    uflows_batch = torch.stack(flows) if any flow else None
    return (7 items)
```

### 每 batch 記憶體操作

```
limg_batch:   128 個 tensor → stack → 192 MB 連續記憶體分配 + 拷貝
uimg_batch:   128 個 tensor → stack → 192 MB
umask_batch:  128 個 tensor → stack → 64 MB
uflows_batch: 128 個 tensor → stack → 4 MB (或 None)
總計: ~450 MB CPU 記憶體分配 + 拷貝 per batch
```

---

## 6. 訓練循環 — `train/train_loop.py:180-282`

```python
for batch in pbar:                    # ← tqdm 包裝 DataLoader
    # CPU→GPU 傳輸
    limg_batch = limg_batch.to(device)    # 192 MB
    uimg_batch = uimg_batch.to(device)    # 192 MB
    umask_batch = umask_batch.to(device)  # 64 MB
    uflows_batch = uflows_batch.to(device) # 4 MB (or None)

    # GPU 前向 (4 次 forward + 1 次 backward)
    student(limg_batch[:,0])              # labeled supervised
    teacher(uimg_batch[:,0])              # teacher t=0
    student(uimg_batch[:,1])              # student t=1
    teacher(uimg_batch[:,1])              # teacher t=1

    # CPU 操作
    update_ema(student, teacher, 0.999)   # ← 20M 參數 Python loop
    loss.item()                           # ← CPU-GPU 同步點
```

---

## 7. P2R Loss（chunked）— `losses/p2rloss.py`

```python
CHUNK_SIZE = 16

def forward(self, dens, seqs, down, ...):
    nonempty_idx = [i for i in range(bs) if seqs[i].size(0) >= 1]
    # 切分 chunk 限制 GPU 記憶體
    for chunk_start in range(0, nb, self.CHUNK_SIZE):  # nb=128 → 8 chunks
        # 每個 chunk: C shape (16, 1, 4096, 500) ≈ 131 MB
        # 每個 chunk: M shape (16, 1, 4096, 500) ≈ 131 MB
        T_chunk, W_chunk = self._process_chunk(...)
```

---

## 8. 卡頓根因分析

### 8.1 DataLoader Index Queue 耗盡（主因）

```
num_workers=8, prefetch_factor=4
初始 index queue = 8 × 4 = 32 個 indices
batch_size = 128 → 需要 128 個結果才能 yield 一個 batch
每個 worker 需貢獻 128/8 = 16 個 samples

時間線:
  t=0:  index queue = [0..31], 8 workers 各取 1 → queue = [8..31]
  t=1:  8 workers 各自處理 1 sample (耗時 ~0.5-1s)
  t=1+: worker 完成 → 結果送入 result queue → 請求下個 index
  t=2:  主程序讀取 result queue, 呼叫 _try_put_indices 補充 index
  ...
  t=16: 主程序收集到 128 個結果, yield batch
         此時 workers 可能已耗盡 index queue → 全部停頓
  t=16+: GPU 處理 batch (~2-3s)
          8 workers 停頓等待主程序補充 index
  t=19+: 主程序 GPU 工作完成, 請求下個 batch
         開始補充 index queue → workers 重新啟動
```

**關鍵問題**: `prefetch_factor=4` 導致只有 `8×4=32` 個初始 indices，遠小於 batch_size=128。Worker 在 GPU 運算期間耗盡 index queue 後集體停頓。

### 8.2 為什麼是每 6 batch？

```
GPU 每批運算時間 ≈ 2-3s (4 forward + 1 backward + P2R loss)
8 workers 每批消耗 index ≈ 128 + (GPU 時間內 worker 完成的額外 samples)
以每 worker 0.5s/sample 計, GPU 時間內可完成 4-6 sample/worker = 32-48 indices

每次 batch 結束時, index queue 剩餘 ≈ 32 - (128-32) + 補充量 - 消耗量 ≈ 0
約每 6 batch 一次 timing 對齊, 所有 worker 同時耗盡 index queue, 導致明顯卡頓
```

### 8.3 次要因素

| 因素 | 位置 | 影響 |
|------|------|------|
| collate_fn stack 4次 | fdst.py:275-289 | ~450MB CPU 記憶體分配, ~200ms |
| .to(device) 4次 | train_loop.py:193-200 | ~450MB PCIe 傳輸, ~30ms |
| update_ema Python loop | train_loop.py:272 | 20M 參數 mul_/add_, ~5-10ms |
| generate_points loop | train_loop.py:250 | for b in range(128) + nonzero |
| PIL JPEG 解碼 512 張 | fdst.py:_load_image | 每 batch 最大 I/O 開銷 |

---

## 9. 修復方向

### 9.1 增加 prefetch_factor

```python
# datasets/__init__.py
# 確保 prefetch 數量 >= batch_size, 避免 worker 空等
effective_prefetch = max(4, batch_size // num_workers * 2)
prefetch_factor = effective_prefetch if num_workers > 0 else 2

# batch_size=128, num_workers=8 → effective_prefetch = 32
# 初始 index queue = 8 × 32 = 256 >> batch_size=128
# Worker 永遠不會耗盡 index queue
```

### 9.2 調整建議

```
batch_size=128, num_workers=8:
  目前: prefetch_factor=4 → queue=32 → 瓶頸
  修正: prefetch_factor=32 → queue=256 → 充裕
```
