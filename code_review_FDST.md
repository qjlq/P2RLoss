# FDST 相關代碼審查報告

> 審查範圍：P2RLoss 專案中所有與 FDST 相關的代碼，包含 datasets、losses、models、train loop、preprocess、main 等模組。

---

## 專案文件結構

```
P2RLoss/
├── config.py                     # 配置 (yacs CfgNode)
├── logger.py                     # 日誌
├── lr_scheduler.py               # 學習率排程
├── main.py                       # 主入口：訓練/eval
├── utils.py                      # checkpoint、繪圖、seed 工具
├── run.sh                        # 啟動腳本 (SHA 實驗)
├── datasets/
│   ├── __init__.py               # DataLoader 工廠
│   ├── fdst.py                   # FDST 資料集 (影片序列)
│   ├── shha.py                   # ShanghaiTech 資料集
│   └── utils.py                  # NormalSample 增強/crop
├── losses/
│   ├── __init__.py               # loss 工廠
│   └── p2rloss.py                # P2R loss
├── models/
│   ├── __init__.py               # model 工廠
│   ├── vgg16bn.py                # VGG16_BN 骨幹 + ConvGRU
│   └── utils.py                  # UpSample_P2P、SimpleDecoder、ConvGRU
├── train/
│   └── train_loop.py             # 訓練迴圈 (BPTT、EMA、flow warping)
├── preprocess/
│   └── preprocess_fdst.py        # FDST 預處理 (VIA JSON→npy、flow 計算)
└── exp/
    ├── fdst-down2/               # 完整實驗 (100 epochs)
    ├── fdst-down2_20260709_142036/
    ├── fdst-down2_20260709_142813/
    ├── fdst-down2_20260709_150111/
    ├── fdst-down2_20260709_150215/
    └── fdst-down2_20260709_153322/
```

---

## 疑點代碼片段

### 1. 🔴 `main.py:174-178` — save_checkpoint 在更新 max_accuracy 前呼叫

```python
# main.py:174-178
            if mae * 4 + mse < max_accuracy[0] * 4 + max_accuracy[1]:
                save_checkpoint(
                    config, f"best_epoch{epoch}", [teacher, student],
                    optimizer, lr_scheduler, scaler, max_accuracy, logger)  # ← 傳入舊的 max_accuracy
                max_accuracy = (mae, mse, loss)  # ← 更新在 save 之後
```

**問題**：`save_checkpoint` 寫入 checkpoint 時使用的是**尚未更新的** `max_accuracy`。若這一輪是最佳 epoch，checkpoint 內存的卻是上一輪的 best，而非當前最佳。

**影響**：load 回來繼續訓練時，best 比較的基準仍是舊值，相當於最佳 epoch 的紀錄丟失了。

---

### 2. 🔴 `datasets/fdst.py:196` — _load_annotation 只嘗試第一個目錄

```python
# datasets/fdst.py:101-104
        self.dot_dirs = (
            os.path.join(root_path, mode + '_data', 'new-anno'),
            os.path.join(root_path, mode + '_data', 'annotations'),
        )

# datasets/fdst.py:195-197
    def _load_annotation(self, vid, frm):
        p = os.path.join(self.dot_dirs[0], f"GT_{vid}_{frm}.npy" if vid is not None else f"GT_{frm}.npy")
        return torch.from_numpy(np.load(p, mmap_mode='r').copy())[:, :2]
```

**問題**：`self.dot_dirs` 被定義為 `('new-anno', 'annotations')`，但 `_load_annotation` 永遠只用 `self.dot_dirs[0]`，從不 fallback 到 `self.dot_dirs[1]`。如果 `new-anno` 不存在而 `annotations` 存在，會直接 crash。

---

### 3. 🟡 `main.py:292-347` — validate 函數內有 dead code

```python
# main.py:291-292
    return mae_meter.avg, mse_meter.avg ** 0.5, loss_meter.avg
    model.eval()  # ← line 292 開始全是 dead code

    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    mae_meter = AverageMeter()
    mse_meter = AverageMeter()

    end = time.time()
    for idx, batch in enumerate(data_loader):
        ...
    return mae_meter.avg, mse_meter.avg ** 0.5, loss_meter.avg  # line 347
```

**問題**：line 291 的 `return` 之後還有 56 行代碼（舊版 validate 實作），完全不會被執行。這會造成維護混淆——修改時可能改到 dead code 卻不自知。

---

### 4. 🟡 `train/train_loop.py:202` — uimg_batch.shape 解包時若 None 會 crash

```python
# train/train_loop.py:181-202
        if len(batch) >= 7:
            limg_batch, lseqs, lids, uimg_batch, umask_batch, uids, uflows_batch = batch
        elif len(batch) == 3:
            limg_batch, lseqs, lids = batch
            uimg_batch = None          # ← 設為 None
            umask_batch = None
            uflows_batch = None
        else:
            raise RuntimeError("Unsupported batch format from dataloader")
        ...
        B, T, C, H, W = uimg_batch.shape  # ← line 202，若 uimg_batch=None 則 AttributeError
```

**問題**：當 batch 長度為 3 時（`len(batch) == 3`），`uimg_batch` 被設為 `None`，但 line 202 無條件對其解包。目前 FDST 訓練資料永遠返回 7 元素因此不會觸發，但若日後載入格式改變的 checkpoint 或使用 eval-only 模式可能爆炸。

---

### 5. 🟢 `datasets/__init__.py:32-35` — 混亂的 config 屬性訪問

```python
# datasets/__init__.py:32-35
    seq_len = getattr(config.DATA, 'SEQ_LEN', 1) if hasattr(config, 'DATA') else getattr(config, 'SEQ_LEN', 1)
    seq_stride = getattr(config.DATA, 'SEQ_STRIDE', 1) if hasattr(config, 'DATA') else getattr(config, 'SEQ_STRIDE', 1)
    flow_root = getattr(config.DATA, 'FLOW_ROOT', None) if hasattr(config, 'DATA') else None
    flow_ext = getattr(config.DATA, 'FLOW_EXT', '.npy') if hasattr(config, 'DATA') else '.npy'
```

**問題**：`hasattr(config, 'DATA')` 永遠為 True（`_C` 內定義了 `DATA`），else 分支全部是 dead code。同時 `config.DATA.LABEL_PERCENT` 和 `LABEL_PROTOCOL` 沒有在 `_C.DATA` 中定義預設值，但 `config.py:131-134` 是透過 `config.defrost()` 動態新增的——依賴 CLI arg 的傳遞順序。

---

### 6. 🟢 `main.py:90` 搭配 `config.py:72` — LR scheduler step_size 過大

```python
# main.py:90
    lr_scheduler = StepLR(optimizer, step_size=config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS, gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE)
```

```python
# config.py:72
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 3500
```

**問題**：`DECAY_EPOCHS = 3500`，而 FDST 訓練僅 100 epochs，SHA 訓練僅 50 epochs。學習率在整個訓練過程中**從未衰減**。若預期行為是固定 LR 則無問題；否則應根據實際總 epoch 調整。

---

### 7. 🟢 `datasets/__init__.py:19-48` — build_loader 的 pin_memory / persistent_workers 設定

```python
# datasets/__init__.py:39-48
    return DataLoader(
        data_set,
        batch_size = batch_size if (mode == 'train') else 1,
        num_workers = num_workers,
        pin_memory = True if (mode == 'train') else False,
        shuffle = (mode == 'train'),
        collate_fn=Dataset.collate_fn,
        persistent_workers = num_workers > 0,
        prefetch_factor = 4 if num_workers > 0 else 2,
    )
```

**問題**：`pin_memory=True` 在訓練模式下啟用，但 `num_workers > 0` 時 `persistent_workers=True`。每個 epoch 結束時 DataLoader 不會重建 worker processes，節省了創建 overhead——這是好的。但 `prefetch_factor=4` 在 `num_workers=0` 時仍傳入 `2`，實際上當 `num_workers=0` 時 `prefetch_factor` 無效。無功能性影響。

---

### 8. 🟢 `datasets/fdst.py:58` — has_subdirs 檢測可能在特殊目錄結構下誤判

```python
# datasets/fdst.py:56-58
        entries = [e for e in os.listdir(images_dir) if not e.startswith('.')]
        has_subdirs = any(os.path.isdir(os.path.join(images_dir, e)) for e in entries)
```

**問題**：若 images_dir 同時含有子目錄和 .jpg 檔案（例如 FDST 標準結構 `images/seq1/`, `images/seq2/` 等），`has_subdirs` 正確為 True。但若 images_dir 中混有非影片的子目錄（如 `.DS_Store` 雖然被過濾，但若有用戶創建的子目錄），則會被誤認為影片目錄。通常 FDST 資料格式固定，影響有限。

---

## 摘要

| 嚴重程度 | 檔案 | 行數 | 描述 |
|----------|------|------|------|
| 🔴 高 | `main.py` | 174-178 | save_checkpoint 在新 best 確認前就已寫入 |
| 🔴 高 | `datasets/fdst.py` | 196 | `_load_annotation` 不嘗試 fallback 目錄 |
| 🟡 中 | `main.py` | 292-347 | 56 行 dead code（舊版 validate） |
| 🟡 中 | `train/train_loop.py` | 202 | `uimg_batch` 解包未防 None |
| 🟢 低 | `datasets/__init__.py` | 32-35 | else 分支 dead code |
| 🟢 低 | `config.py` + `main.py` | 72 + 90 | `DECAY_EPOCHS=3500`，LR 在訓練期間從不衰減 |
| 🟢 低 | `datasets/__init__.py` | 39-48 | prefetch_factor 在 num_workers=0 時無效 |
| 🟢 低 | `datasets/fdst.py` | 58 | has_subdirs 邏輯對非影片目錄容錯性低 |
