# P2R 論文分析與代碼比對驗證報告

> 比對對象：P2R 原著論文結論 vs `P2RLoss` 實際代碼實現
> 審查日期：2026-07-17

---

## 1. 反 Sigmoid 函數的極端懲罰

### 論文聲稱

> P2R 使用反 Sigmoid 函數 $\mathcal{S}(p) = -\log(1/p - 1)$ 計算分數成本，該函數在 $p \to 0$ 或 $p \to 1$ 時梯度極度陡峭，會放大空間錯位導致的 Loss 引發 V 型暴衝。

### 代碼驗證：❌ 不存在

**`losses/p2rloss.py:73`** — 成本矩陣計算的核心程式碼：

```python
C = C * self.cost_point - A_chunk.view(nb, 1, HW, 1) * self.cost_class
```

| 組件 | 實現方式 | 說明 |
|------|---------|------|
| 距離成本 `C` | `torch.norm(x_col - y_row, dim=-1)` | L2 歐氏距離 |
| 分類成本 `A_chunk` | **原始 logits**（未經 sigmoid） | 直接使用密度 logits |
| 最終 loss | `F.binary_cross_entropy_with_logits(A, T_full, weight=W_full)` | BCE + logits 內建 sigmoid |

**關鍵發現**：此實現在成本矩陣中直接使用原始 logits（`A_chunk`），完全沒有論文描述的反 Sigmoid 函數。`BCEWithLogitsLoss` 雖然內部會對 logits 做 sigmoid，但那是計算最終 loss 時的事，與成本矩陣無關。

**V 型暴衝的真實原因**（來自代碼審查）：`max_radis` 未隨 `down` 縮放 → 正樣本匹配區域縮小 4 倍 → 正樣本極度稀缺 → BCE loss 中正負樣本失衡 → 模型暴增權重補償 → 梯度爆炸。此修復已在 `p2rloss.py:44-46` 完成。

---

## 2. 網格狀異常與過度激活 (Over-activation)

### 論文聲稱

> 損失函數設定不當會導致前景周圍像素被過度激活（over-activated），解碼器將周圍所有像素都識別為前景 → 網格狀輸出。

### 代碼驗證：🟡 部分吻合

**`losses/p2rloss.py:151-154`** — 最終 loss 計算：

```python
loss = tF.binary_cross_entropy_with_logits(A, T_full, weight=W_full, reduction='mean')
```

此 loss 確實對**所有像素**（無論正負樣本）計算梯度，沒有阻擋背景梯度。但以下因素可能加劇 over-activation：

| 因素 | 代碼位置 | 現狀 |
|------|---------|------|
| `pos_weight` 過高 | `p2rloss.py:86-88` | 預設 20.0（正樣本權重遠高於背景） |
| `max_radis` 縮水 | `p2rloss.py:66` | 已被修復（動態縮放） |
| 無 `strong_aug` | `datasets/utils.py:31-35` | `strong_aug`（含 ColorJitter）已定義但**從未被任何程式碼調用** |

**`strong_aug` 死程式碼驗證**：

```python
# datasets/utils.py:31-35 — 定義了但從未被呼叫
self.strong_aug = transforms.Compose([
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.25),
    self.im2tensor
])

# datasets/utils.py:74-92 — process_unlabel 完全未使用 strong_aug
def process_unlabel(self, image):
    if self.train:
        images = self.crop_and_resize(image)  # ← 直接 crop_and_resize
    ...
```

---

## 3. Stage 1 訓練長度與策略

### 論文聲稱

> P2R 訓練 1500 epochs，前 100 epochs 僅用有監督數據初始化，無標籤權重 $\alpha$ 從 0 緩慢遞增。

### 代碼驗證：🟡 方向正確但差距大

| 參數 | 論文 | 當前專案 | 差異 |
|------|------|---------|------|
| 總 epochs | 1500 | **50** (預設) | 30× 差距 |
| Stage 1 長度 | 100 | **25** | 4× 差距 |
| $\alpha$ warmup | 從 0 緩增 | **瞬間從 0 切到 0.999** | 硬切換 vs 緩啟動 |

**硬切換驗證**（`train/train_loop.py`）：

```python
# 第 25 epoch 前：stage='sup'，完全不用 teacher
# 第 25 epoch：瞬間開啟 teacher + EMA=0.999 + 偽標籤
# 無任何過渡期或 α 遞增機制
```

---

## 4. 梯度裁剪

### 代碼驗證：❌ 定義了但從未使用

```python
# config.py:75
_C.TRAIN.CLIP_GRAD = 5.0
```

`CLIP_GRAD=5.0` 僅存在於 config，**在任何訓練程式碼（`train_loop.py` / `main.py`）中都未找到 `clip_grad_norm_()` 的調用**。梯度爆炸時沒有任何防護。

---

## 5. `drop_path` 設定

### 代碼驗證

| 參數 | 當前值 | 建議值（論文） |
|------|--------|--------------|
| `EMAC_DROP_PATH` | **0.3** | 0.5-0.8（ViT 在小數據集） |

---

## 總結：論文分析正確性評分

| 論文觀點 | 驗證結果 | 說明 |
|---------|---------|------|
| 反 Sigmoid 函數導致 V 型暴衝 | ❌ 不存在 | 此實作使用原始 logits + BCEWithLogits，無反 Sigmoid |
| Over-activation 導致網格輸出 | 🟡 部分吻合 | pos_weight=20 加劇正樣本權重失衡 |
| 需要長 Stage 1（100+ epochs） | ✅ 正確 | 當前 25 epochs 嚴重不足 |
| 需要 $\alpha$ 緩慢遞增 | ✅ 正確 | 當前硬切換，無過渡 |
| 需要梯度裁剪 | ✅ 正確 | 已定義 CLIP_GRAD 但未實裝 |
| 需要強數據增強 | ✅ 正確 | `strong_aug` 定義了但從未呼叫 |
| max_radis 動態縮放 | ✅ 已修復 | `p2rloss.py:44-46` 已完成 |

### 優先修復清單（按影響程度）

1. **實裝梯度裁剪**：`torch.nn.utils.clip_grad_norm_(student.parameters(), config.TRAIN.CLIP_GRAD)` — 10 分鐘
2. **啟用 `strong_aug`**：在 `process_unlabel` 中加入強增強管線 — 30 分鐘
3. **Stage 1 延長至 50-100 epochs**：修改 `STAGE_1` 或訓練腳本 — 1 行
4. **EMA $\alpha$ 緩慢遞增**：從 0.9 線性遞增至 0.999 over 25 epochs — 2 小時
