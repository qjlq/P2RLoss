# FDST 項目光流數據流水線文檔

> 本文檔記錄 FDST 數據集中光流數據的產生、載入與使用的完整流水線，涵蓋 RAFT 外部光流與內置 PWC-Net 兩個分支。

---

## 目錄

1. [總覽：兩種光流分支](#1-總覽兩種光流分支)
2. [分支 A：外部 RAFT 光流](#2-分支-a外部-raft-光流)
3. [分支 B：內置 PWC-Net 光流](#3-分支-b內置-pwc-net-光流)
4. [解析度對齊](#4-解析度對齊)
5. [分支切換控制](#5-分支切換控制)

---

## 1. 總覽：兩種光流分支

```
                    ┌─────────────────┐
                    │  FDST 影像數據   │
                    │  (B, T, C, H, W)│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   train_loop.py  │
                    │  半監督分發點    │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
     ┌────────────────┐ ┌──────────┐ ┌──────────────┐
     │ VGG16BN 分支   │ │EMAC 分支 │ │EMAC 分支     │
     │ (外部 RAFT)    │ │(內置PWC) │ │(外部 flow)   │
     │ 已實現         │ │ 已實現   │ │ 可選         │
     └────────────────┘ └──────────┘ └──────────────┘
```

| 分支 | 模型 | Flow 來源 | 現狀 |
|------|------|-----------|------|
| **A** | VGG16BN | 外部 `.npy`（RAFT/Farneback） | ✅ 已實現，使用 `uflows_batch` |
| **B** | EMAC | 內置 PWC-Net | ✅ 已實現，使用 `EMACWrapper.compute_flow()` |
| **C** | EMAC | 外部 `.npy`（RAFT） | 🔀 可選，傳入 `flow=` 參數時使用 |

---

## 2. 分支 A：外部 RAFT 光流

### 2.1 Flow 產生（離線預處理）

**執行命令**：
```bash
python /media/SSD/OSshareSpace/localSpace/workPlace/zhongKe/gd/FDST/raft_precomputed.py \
    --root /media/SSD/OSshareSpace/localSpace/workPlace/zhongKe/gd/FDST \
    --down 2
```

**核心代碼** (`raft_precomputed.py`):

| 環節 | 代碼 | 說明 |
|------|------|------|
| 模型初始化 | `raft_large(weights=Raft_Large_Weights.DEFAULT)` | 預訓練 RAFT 大模型 |
| 影像預處理 | `preprocess_frame(frame, scale_factor=0.5)` | 縮放 50%、8 倍數對齊、歸一化至 [-1,1] |
| 前向推論 | `list_of_flows = model(img1, img2)` | RAFT 多尺度迭代 |
| 取最後輸出 | `predicted_flow = list_of_flows[-1]` | 最後一次迭代的 refine 結果 |
| 解析度轉換 | `target_h = orig_h // down`, `target_w = orig_w // down` | 縮放至目標解析度 |
| 位移補償 | `predicted_flow *= (target_w / model_w, target_h / model_h)` | 保持物理位移量正確 |
| 儲存格式 | `flow_numpy.squeeze(0).cpu().numpy()` | `(2, H_out, W_out)` CHW, float32 |
| 檔名規則 | `{video_id}_{frame_name}_flow.npy` | 與 Farneback 腳本對齊 |

### 2.2 Flow 載入（訓練時）

**`datasets/fdst.py:_load_flow_for_sequence`** (L215-240):

```python
def _load_flow_for_sequence(self, list_ref, start_idx):
    if not self.flow_root:
        return None
    idxs = self._get_sequence_indices(list_ref, start_idx)
    flows = []
    for i in range(len(idxs) - 1):
        vid, frm = list_ref[idxs[i]]
        fn = os.path.join(self.flow_root, f"{vid}_{frm}_flow{self.flow_ext}" 
                          if vid is not None else f"{frm}_flow{self.flow_ext}")
        arr = np.load(fn, mmap_mode='r')
        flows.append(torch.from_numpy(arr).float())
    # 堆疊為 (T-1, 2, H, W)
    return torch.stack(flows, dim=0)
```

**`datasets/fdst.py:collate_fn`** (L255-274):

```python
# 打包為 batch: (B, T-1, 2, H, W) 或 None
uflows_batch = torch.stack(stacked_flows, dim=0)
```

### 2.3 Flow 在 VGG16BN 訓練中的使用

**`train/train_loop.py`** VGG16BN 半監督分支 (L373-427):

```python
# 1. 按時間步取出 flow
flow_t_minus = uflows_batch[:, t - 1]  # (B, 2, H, W)

# 2. Teacher 預測 → 提取偽點
with torch.no_grad():
    points_prev_list = generate_points_from_density(
        teacher_pred_prev, threshold=pseudo_thresh
    )

# 3. 用 flow 將偽點 warp 到當前幀
warped_points_list = warp_points_with_flow(
    points_prev_list, flow_t_minus,
    flow_down=2,         # RAFT 以 --down 2 提取
    model_down=down      # 模型下採樣倍數
)

# 4. Student 在 warp 後的偽點上計算 P2R loss
loss_p2r = p2r(student_pred_t, seqs_for_loss, down=down)
```

**`sample_flow_at_points`** (L90-120):

```python
def sample_flow_at_points(flow, points, flow_down=2, model_down=4):
    scale = model_down / flow_down  # 密度圖 → flow 空間轉換 = 2
    xs = pts[:, 0] * scale
    ys = pts[:, 1] * scale
    nx = (xs / float(max(W - 1, 1))) * 2.0 - 1.0   # 歸一化到 flow 空間 [-1,1]
    ny = (ys / float(max(H - 1, 1))) * 2.0 - 1.0
    sampled = F.grid_sample(flow, grid.view(1, -1, 1, 2), align_corners=True)
```

**`warp_points_with_flow`** (L123-158):

```python
def warp_points_with_flow(points_list, flow_batch, flow_down=2, model_down=4):
    scale = model_down / flow_down       # 密度→flow
    inv_scale = flow_down / model_down   # flow→密度
    pts_flow = points * scale            # 轉到 flow 空間做 warp
    warped_flow = pts_flow + flow_vecs
    warped_pts = warped_flow * inv_scale # 轉回密度圖空間
```

---

## 3. 分支 B：內置 PWC-Net 光流

### 3.1 PWC-Net 整合

**`models/emac_wrapper.py`** 中的 `compute_flow` 方法 (L209-228):

```python
def compute_flow(self, img_prev, img_cur):
    """End-to-end PWC-Net flow from prev → cur."""
    B, C, H, W = img_cur.shape
    permute_bgr = [2, 1, 0]

    # ImageNet 解歸一化 + BGR 轉換 + 縮放至 [0,1]
    img_all = torch.cat([
        denormalize(img_cur)[:, permute_bgr] / 255.0,
        denormalize(img_prev)[:, permute_bgr] / 255.0,
    ], dim=1).to(img_cur.device)

    # PWC-Net 前向
    flo_raw = self.emac.pwc(img_all)          # (B, 2, H/4, W/4)
    flo_shape = flo_raw.shape[-2:]

    # 放大到原始解析度，位移數值補償
    flo = F.interpolate(flo_raw, (H, W), mode='bilinear', align_corners=False)
    flo = flo * (flo_shape[0] * flo_shape[1]) / (H * W)
    return flo
```

### 3.2 EMACWrapper.forward 中的 temporal fusion (L311-327)

```python
# Temporal fusion with PWC-Net or precomputed flow
if flow is not None:
    # 外部 flow (RAFT)：resize 後 warp，不返回 flo
    flow_resized = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
    scale = torch.tensor([...], device=flow.device)
    flow_resized = flow_resized * scale.view(1, 2, 1, 1)
    pred_prev_warp = warp(pred_prev, flow_resized.detach())
    flo = None
else:
    # 內置 PWC-Net：compute_flow → warp → 返回 flo
    flo = self.compute_flow(img_template, img_current)
    pred_prev_warp = warp(pred_prev, flo.detach())

pred_fuse = self._fuse_dense(pred_prev_warp, pred_cur)   # TransFuse 融合
```

### 3.3 訓練循環中的使用 (L319-324)

```python
# EMAC 分支完全忽略 uflows_batch
with autocast(enabled=amp_enabled):
    student_pred_t, flo_t, pred_prev_warp_raw, pred_cur_raw = student(
        frame_t, templates=[uimg_batch[:, (t + 1) % T]],
        return_aux=True   # 返回 (pred, flow, prev_warp, cur_pred)
    )
```

### 3.4 光流重建損失 + TV 平滑 (L340-358)

```python
# RGB 光流重建損失 (MSE on warped RGB)
perm_bgr = [2, 1, 0]
cur_rgb = denormalize(frame_t_teacher)[:, perm_bgr] / 255.0
prev_rgb = denormalize(uimg_batch_teacher[:, (t + 1) % T])[:, perm_bgr] / 255.0
img_warp = emac_warp(prev_rgb, flo_t)
opt_loss = F.mse_loss(img_warp, cur_rgb) * 0.05

# Flow 平滑正則 (TV)
flow_tv = tv_loss(flo_t) * 0.01

loss_total = loss_p2r + sup_loss + opt_loss + flow_tv
```

### 3.5 PWC-Net 的 Correlation 降級（`models/emac_wrapper.py`）

```python
class _CorrelationPyTorch(nn.Module):
    """使用 F.unfold + einsum 替代 CUDA correlation kernel"""
    def forward(self, f1, f2):
        B, C, H, W = f1.shape
        D = self.max_displacement
        kW = 2 * D + 1
        f2_unfold = F.unfold(f2, kernel_size=(kW, kW), 
                             padding=D, stride=self.stride2)
        f2_unfold = f2_unfold.view(B, C, kW * kW, H * W)
        f1_flat = f1.view(B, C, H * W).unsqueeze(2)
        corr = (f1_flat * f2_unfold).sum(dim=1)
        return corr.view(B, kW * kW, H, W)
```

---

## 4. 解析度對齊

| 層級 | 解析度 | 倍數 | 控制參數 |
|------|--------|------|----------|
| 原始影像 | `H_img × W_img` | 1× | FDST 原始解析度 |
| RAFT flow (down=2) | `H_img//2 × W_img//2` | 0.5× | `raft_precomputed.py:--down 2` |
| PWC-Net 輸出 | `H_img//4 × W_img//4` | 0.25× | PWCNet 網絡結構 (5次 2倍下採樣) |
| 密度圖 (EMAC) | `H_img × W_img` | 1× | SpatialOutputAdapter 還原 |
| 模型輸入 (crop) | `256 × 256` | 固定 | `datasets/utils.py:21` |

**Warp 時兩分支的解析度轉換**：

| 分支 | Warp 操作 | 空間單位 |
|------|----------|----------|
| VGG16BN + RAFT | `warp_points_with_flow(flow_down=2, model_down=4)` | 密度圖解析度: `(256//4, 256//4)` |
| EMAC + PWC-Net | `warp(pred_prev, flo.detach())` | 全解析度: `(256, 256)` |

---

## 5. 分支切換控制

### 模型啟動命令

| 分支 | `MODEL.NAME` | Data flow |
|------|-------------|-----------|
| VGG16BN + 外部 RAFT | `vgg16bn` | `uflows_batch` via `FLOW_ROOT` |
| EMAC + 內置 PWC-Net | `emac` | `student(frame, templates=[...], return_aux=True)` |

### train_loop 中的分支判斷 (L267)

```python
is_emac = (model_name == 'emac')

if is_emac:
    # EMAC 分支：用 PWC-Net，忽略 uflows_batch
    student_pred_t, flo_t, ... = student(..., return_aux=True)
else:
    # VGG16BN 分支：用外部 flow
    flow_t_minus = uflows_batch[:, t - 1]
    ...
```

### EMACWrapper 中的分支判斷 (L312-321)

```python
if flow is not None:
    # 外部 flow（可選，傳入 flow= 參數時啟用）
    pred_prev_warp = warp(pred_prev, flow_resized.detach())
    flo = None
else:
    # 預設：內置 PWC-Net
    flo = self.compute_flow(img_template, img_current)
    pred_prev_warp = warp(pred_prev, flo.detach())
```
