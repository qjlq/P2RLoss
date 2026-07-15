# FDST Temporal Crowd Counting — 模型替換參考文檔

> 目的：完整記錄 FDST 時序人群計數訓練的程式碼流水線，聚焦於將 VGG16 替換為其他 backbone 所需了解的所有接口、依賴關係與修改點。

---

## 專案文件結構

```
P2RLoss/
├── config.py                     # 配置管理 (yacs CfgNode)
├── main.py                       # 主入口：訓練/eval 流程
├── p2r_utils.py                  # checkpoint、繪圖、seed 工具
├── logger.py                     # 日誌
├── models/
│   ├── __init__.py               # build_model 工廠 ← 模型替換處
│   ├── vgg16bn.py                # VGG16_BN — 被替換的核心檔案
│   ├── utils.py                  # 共用組件 (可復用)
│   └── emac_wrapper.py           # E-MAC ViT 包裝器 (新模型參考)
├── losses/
│   ├── __init__.py               # build_loss 工廠 (與 backbone 無關)
│   └── p2rloss.py                # P2R Loss (與 backbone 無關)
├── train/
│   └── train_loop.py             # 訓練迴圈 (BPTT, EMA, flow warping)
├── datasets/
│   ├── __init__.py               # build_loader 工廠 (與 backbone 無關)
│   ├── fdst.py                   # FDST 資料集
│   ├── shha.py                   # SHHA 資料集
│   └── utils.py                  # NormalSample 增強
└── preprocess/
    └── preprocess_fdst.py        # FDST 預處理 (與 backbone 無關)
```

---

## 1. `models/vgg16bn.py` — 被替換的目標模型

### 類定義與 forward 接口

```python
class VGG16_BN(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        vgg = models.vgg16_bn(pretrained=True)
        features = list(vgg.features.children())
        lids = [0, 33, 43]
        self.encoders = nn.ModuleList(
            nn.Sequential(*features[a:b]) for a, b in zip(lids[:-1], lids[1:])
        )                           # 2 級 encoder，輸出通道均為 512
        self.num_channels = [512, 512]
        self.fuse_layer = UpSample_P2P(self.num_channels, ouc=256, bn=False, relu=False)
        self.decoders = SimpleDecoder(in_channel=256, fea_channel=256,
                                       up_scale=2, out_channel=1)  # → 1ch density
        self.temporal_in_channels = 1
        self.temporal_hidden = 32
        self.temporal = TemporalUnit(mode='convgru', in_ch=1, hid_ch=32, kernel=3)
        self.to_temporal = nn.Conv2d(1, 1, kernel_size=1)
        self.hidden2logit = nn.Conv2d(32, 1, kernel_size=1)
        self.down = 4   # ← 新模型須設置正確的下採樣倍數

    def forward(self, image, prev_h=None, need_fp=False):
        """
        image:  (B, 3, H, W)
        prev_h: None 或 (B, 32, H_out, W_out) — ConvGRU hidden state
        returns:
            pred_logits: (B, 1, H_out, W_out)
            next_h:      (B, 32, H_out, W_out)
        """
        fea2 = self.encoding(image)     # 多尺度特徵提取
        denmap = self.decoding(fea2)    # → (B, 1, H_out, W_out)
        x_t = self.to_temporal(denmap)
        next_h = self.temporal(x_t, prev_h)
        pred_logits = self.hidden2logit(next_h)
        return pred_logits, next_h

    def init_hidden(self, batch_size, device, spatial_size):
        return torch.zeros(batch_size, 32, H, W, device=device)
```

### 新模型必須實作的接口契約

```python
class NewBackbone(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.down = 4  # density map 相對於原圖的縮放比

    def forward(self, image, prev_h=None, need_fp=False):
        """
        image:      (B, 3, H, W)
        prev_h:     None 或 (B, hid_ch, H_out, W_out) — 前一幀 hidden state
        need_fp:    是否在前向中應用 dropout
        returns:
            pred_logits: (B, 1, H_out, W_out) — 密度圖 logits
            next_h:      (B, hid_ch, H_out, W_out) — 下一幀的 hidden state
        """
        raise NotImplementedError

    def init_hidden(self, batch_size, device, spatial_size):
        """
        spatial_size: (H_out, W_out) — 密度圖的空間尺寸
        returns: (B, hid_ch, H_out, W_out)
        """
        raise NotImplementedError
```

---

## 2. `models/utils.py` — 可復用的共用組件

```python
class UpSample_P2P(nn.Module):
    """多尺度特徵融合。
    incs: list[int] — 各 encoder stage 的通道數
    ouc:  int       — 融合後輸出通道
    forward(xs): xs 為 list[Tensor(B, C_i, H_i, W_i)]
                 返回 (B, ouc, H_out, W_out) 所有特徵上探樣至同一解析度後相加 + fuse
    """
    def __init__(self, incs, ouc, bn=True, relu=True): ...

class SimpleDecoder(nn.Sequential):
    """密度圖解碼器: in_ch → fea_ch → out_ch * up_scale² → PixelShuffle"""
    def __init__(self, in_channel=128, fea_channel=64, up_scale=1, out_channel=1): ...

class ConvGRUCell(nn.Module):
    """標準 ConvGRU 單元: z (update), r (reset) gates"""
    def __init__(self, in_ch, hid_ch, kernel_size=3): ...
    def forward(self, x, h):
        """x: (B, in_ch, H, W), h: (B, hid_ch, H, W) → (B, hid_ch, H, W)"""

class TemporalUnit(nn.Module):
    """時序單元包裝器，目前實現 ConvGRU 模式"""
    def __init__(self, mode='convgru', in_ch=1, hid_ch=32, kernel_size=3): ...
    def forward(self, x, h):
        """x: (B, in_ch, H, W), h: (B, hid_ch, H, W) or None → (B, hid_ch, H, W)"""
```

### UpSample_P2P 使用示例

新模型若有多級 encoder 輸出，可以用 `UpSample_P2P` 將它們融合：
```python
# 假設新模型有 3 級 encoder 輸出，通道數分別為 [256, 512, 1024]
self.fuse_layer = UpSample_P2P([256, 512, 1024], ouc=256, bn=False, relu=False)
fea = self.fuse_layer([fea1, fea2, fea3])  # 自動上探樣到 fea1 解析度後融合
```

---

## 3. `models/__init__.py` — 模型工廠

```python
from .vgg16bn import VGG16_BN

def build_model(config):
    model = {
        'vgg16bn': VGG16_BN       # ← 加入新模型映射，如 'resnet50': ResNet50
    }[config.NAME.lower()]

    return model(config), model(config)  # 回傳 (student, teacher)
```

### 接入新模型的方式

```python
from .resnet_backbone import ResNetBackbone  # 新 backbone

def build_model(config):
    model_cls = {
        'vgg16bn': VGG16_BN,
        'resnet50': ResNetBackbone,   # ← 加入
    }[config.NAME.lower()]
    return model_cls(config), model_cls(config)
```

啟動指令：
```bash
python main.py ... --opts MODEL.NAME resnet50 ...
```

---

## 4. `config.py` — 配置字段

```python
# 模型相關
_C.MODEL = CN()
_C.MODEL.NAME = 'VGG16BN'      # 字串，對應 models/__init__.py 的 dict key
_C.MODEL.RESUME = ''            # resume checkpoint 路徑
_C.MODEL.FACTOR = 1             # 傳給 loss（可忽略）
_C.MODEL.LOSS = 'P2R'           # 損失函數選擇

# 訓練相關
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 50
_C.TRAIN.BASE_LR = 5e-5         # 非 backbone 參數的初始 LR
_C.TRAIN.BACKBONE_LR = 1e-5     # backbone (encoders) 參數的 LR
```

---

## 5. `main.py` — 訓練入口與優化器

### 模型建立

```python
student, teacher = build_model(config.MODEL)   # 工廠建立
teacher = copy.deepcopy(student)
for p in teacher.parameters():
    p.requires_grad = False      # teacher 永遠凍結
student.cuda(); teacher.cuda()
```

### 損失函數

```python
criterion, test_criterion = build_loss(config.MODEL)
criterion.cuda(); test_criterion.cuda()
```

### 優化器分組（關鍵改動點）

```python
model_name = config.MODEL.NAME.lower()

# VGG16BN 版本：按 "encoders" 參數名分組
param_dicts = [
    {"params": [p for n, p in student.named_parameters()
                if "encoders" not in n and p.requires_grad]},
    {"params": [p for n, p in student.named_parameters()
                if "encoders" in n and p.requires_grad],
     "lr": config.TRAIN.BACKBONE_LR},
] if model_name == 'vgg16bn' else [
    # 新模型（非 vgg16bn）：全部參數一個分組
    {"params": [p for p in student.parameters() if p.requires_grad]},
]

optimizer = optim.Adam(param_dicts, lr=config.TRAIN.BASE_LR, ...)
```

**若新模型需要 backbone 與非 backbone 不同 LR**，需修改分組規則：
```python
backbone_prefixes = ('layer1', 'layer2', ...)  # 根據新模型參數名稱決定
param_dicts = [
    {"params": [p for n, p in student.named_parameters()
                if not n.startswith(backbone_prefixes) and p.requires_grad]},
    {"params": [p for n, p in student.named_parameters()
                if n.startswith(backbone_prefixes) and p.requires_grad],
     "lr": config.TRAIN.BACKBONE_LR},
]
```

### 訓練循環調用

```python
temporal_train_one_epoch(
    epoch=epoch,
    dataloader=data_loader_train,
    student=student,
    teacher=teacher,
    optimizer=optimizer,
    device=torch.device('cuda'),
    down=getattr(student, 'down', 4),   # ← 使用模型的 self.down 屬性
    ema_alpha=0.999,
    scaler=scaler,
    p2r_loss_cfg=None,
    amp_enabled=True,
    min_batch_size=2,
    stage=stage,
    stage1_epochs=STAGE_1,
    total_epochs=config.TRAIN.EPOCHS,
    model_name=model_name,
    accumulation_steps=accumulation_steps,
)
```

### Validate 函數（與模型類型相關）

```python
@torch.no_grad()
def validate(config, data_loader, model, criterion, ..., model_name='vgg16bn'):
    ...
    if model_name == 'emac':
        # 2-frame sliding window: 每幀用下一幀作模板
        for t in range(T - 1):
            logits_t = model(images[:, t], templates=[images[:, t + 1]])
    else:
        # VGG16BN: 逐幀傳遞 hidden state
        prev_h = None
        for t in range(T):
            logits_t, prev_h = model(images[:, t], prev_h=prev_h)
```

---

## 6. `train/train_loop.py` — 訓練迴圈

### 訓練函數簽名

```python
def train_one_epoch(
    epoch, dataloader, student, teacher, optimizer, device,
    down=4, ema_alpha=0.999, scaler=None, p2r_loss_cfg=None,
    amp_enabled=True, min_batch_size=2, stage='sup',
    stage1_epochs=25, total_epochs=50, model_name='vgg16bn',
    accumulation_steps=1,
):
```

### 監督損失計算（模型調用方式）

```python
if is_emac:
    # EMAC 需兩幀輸入
    lpred = student(lframe, templates=[img_template], density_ref=density_ref)
else:
    # VGG16BN: 標準 forward 接口
    lpred, _ = student(lframe, prev_h=None)
sup_loss = p2r(lpred, ldots, down=down, pos_weight=current_pos_weight)
```

### 半監督分支（VGG16BN 對比）

```python
if is_emac:
    # EMAC 專有邏輯（PWC-Net flow, multi-task loss, ...）
    ...
else:
    # VGG16BN: 標準 ConvGRU BPTT
    student_h = student.init_hidden(B, spatial_size=(H//down, W//down))
    teacher_h = teacher.init_hidden(B, spatial_size=(H//down, W//down))
    for t in range(1, T):
        frame_t = uimg_batch[:, t]
        student_pred_t, student_h = student(frame_t, student_h)   # ← forward
        loss_p2r = p2r(student_pred_t, seqs_for_loss, down=down)
        ...
```

---

## 7. `losses/p2rloss.py` — 損失函數（與 backbone 無關）

```python
class P2RLoss(nn.modules.loss._Loss):
    def forward(self, dens, seqs, down, masks=None, crop_den_masks=None, pos_weight=None):
        """
        dens:  (B, 1, H, W) — 模型輸出的密度 logits
        seqs:  list[B] of (N_i, 2) — GT 點座標 (x, y)，原始圖像解析度
        down:  密度圖 / 原圖 的縮放比（即 self.down）
        pos_weight: 可選，動態正樣本權重
        returns: scalar loss
        """
```

**關鍵**: `down` 參數在 loss 內用於將像素座標映射回原圖空間。新模型的 `self.down` 必須與此一致。

---

## 8. `p2r_utils.py` — Checkpoint 格式

```python
def save_checkpoint(config, epoch, model, optimizer, lr_scheduler, scaler, max_accuracy, logger):
    teacher, student = model
    save_state = {
        'teacher': teacher.state_dict(),       # ← 新模型需相容
        'student': student.state_dict(),       # ← 新模型需相容
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'max_accuracy': max_accuracy,
    }
    torch.save(save_state, save_path)

def load_checkpoint(config, model, optimizer, lr_scheduler, scaler, logger):
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    teacher.load_state_dict(checkpoint['teacher'], strict=False)
    student.load_state_dict(checkpoint['student'], strict=False)
    ...
```

**注意**: 新模型的 `state_dict()` key 名稱必須與 checkpoint 相容，否則 `strict=False` 會跳過不匹配的 layer。

---

## 9. `datasets/fdst.py` — 資料集（與 backbone 無關）

```python
class FDST(data.Dataset):
    def __getitem__(self, index):
        # 訓練時回傳 7 元素:
        return (limg_seq, lseqs, lid, uimg_seq, umask_seq, uid, uflows)
        # limg_seq: (T, C, H, W) — 標籤幀序列
        # uimg_seq: (T, C, H, W) — 無標籤幀序列
        # uflows:   (T-1, 2, H, W) or None

    @staticmethod
    def collate_fn(samples):
        # 打包成 batch: (B, T, C, H, W)
        limg_batch = torch.stack(limg_seqs, dim=0)
```

---

## 10. `datasets/utils.py` — 數據增強（與 backbone 無關）

```python
class NormalSample(object):
    def __init__(self, mean, std, crop_size=(256, 256), resize_factor=0.3, train=False):
        self.half_h, self.half_w = crop_size  # 256×256
        self.im2tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)   # ImageNet 歸一化
        ])

    def process_lable(self, image, dotseq):
        """image: (3, H, W) Tensor, dotseq: (N, 2) → crop → resize → flip → (images, dotseqs)"""
        ...
```

---

## 11. VGG16 替換為新模型的檢查清單

| 步驟 | 檔案 | 需確認事項 |
|------|------|-----------|
| 1 | `models/new_backbone.py` | 實作 `forward(image, prev_h)` → `(logits, next_h)` |
| 2 | `models/new_backbone.py` | 實作 `init_hidden(batch_size, device, spatial_size)` |
| 3 | `models/new_backbone.py` | 設置 `self.down = N`，與 P2R loss 的 `down` 參數一致 |
| 4 | `models/__init__.py` | 加入 import + dict 映射 |
| 5 | `main.py:76-89` | 若新 backbone 參數名稱不含 `encoders`，修改優化器分組條件 |
| 6 | `main.py:260-280` | validate 函數：若新模型不使用 `prev_h`，需調整驗證循環 |
| 7 | `models/new_backbone.py` | 可選用 `UpSample_P2P`、`SimpleDecoder`、`TemporalUnit` 等共用組件 |
| 8 | `train/train_loop.py` | 若有特殊 forward 接口（如 E-MAC 的 templates），需新增條件分支 |
| 9 | 訓練指令 | `--opts MODEL.NAME new_backbone` |

### VGG16 與常見 backbone 的接口對照

| 特性 | VGG16 | ResNet/HRNet 等 |
|------|-------|----------------|
| `forward` 回傳 | `(logits, hidden)` tuple | 必須同樣回傳 `(logits, hidden)` tuple |
| `init_hidden` | 回傳 zero tensor `(B, 32, H, W)` | 可回傳 None（若不使用 ConvGRU），但需修改 train loop |
| `self.down` | 4 | 需根據 backbone 的實際下採樣倍數設定 |
| `encoders` 參數名 | 存在 | 可能沒有 → 需修改優化器分組邏輯 |
| 多尺度特徵 | 2 級 (512, 512) | 可選擇用 `UpSample_P2P` 融合或不使用 |

### 新模型不支援 ConvGRU 時的處理

若新模型不使用 ConvGRU（如 ViT 風格），則：

1. `init_hidden` 回傳 `None`
2. `train_loop.py` 中不調用 `init_hidden`，改為直接 forward
3. `forward` 回傳 `(logits, None)` 或僅 `logits`
4. `main.py` 的 validate 函數中不傳遞 `prev_h`

---

## 12. EMACWrapper — 非 ConvGRU 模型的參考實現

EMACWrapper (`models/emac_wrapper.py`) 展示了非循環（non-recurrent）模型的接入方式，可作為新模型的參考：

```python
class EMACWrapper(nn.Module):
    def __init__(self, config=None):
        self.down = 4

    def forward(self, img_current, templates=None, flow=None, ...):
        # 不需要 prev_h
        return pred_fuse

    def init_hidden(self, batch_size, device, spatial_size):
        return None  # 無 hidden state
```
