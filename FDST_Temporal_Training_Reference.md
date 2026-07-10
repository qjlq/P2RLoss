# FDST 時序人群計數訓練 — 模型替換參考文檔

> 目的：完整紀錄 FDST 時序訓練的程式碼流水線，聚焦於將 VGG16 替換為其他 backbone 所需了解的所有接口與依賴關係。

---

## 專案文件結構

```
P2RLoss/
├── config.py                       # 配置管理 (yacs CfgNode)
├── main.py                         # 主入口：訓練/eval 流程
├── utils.py                        # checkpoint、繪圖、seed 工具
├── logger.py                       # 日誌
├── models/
│   ├── __init__.py                 # build_model 工廠 ← 模型替換處 #1
│   ├── vgg16bn.py                  # VGG16_BN ← 被替換的核心檔案
│   └── utils.py                    # 共用組件 (可復用)
├── losses/
│   ├── __init__.py                 # build_loss 工廠 (與 backbone 無關)
│   └── p2rloss.py                  # P2R Loss (與 backbone 無關)
├── train/
│   └── train_loop.py              # 訓練迴圈 (BPTT, EMA, flow warping)
├── datasets/
│   ├── __init__.py                 # build_loader 工廠 (與 backbone 無關)
│   ├── fdst.py                     # FDST 資料集
│   ├── shha.py                     # SHHA 資料集
│   └── utils.py                    # NormalSample 增強
└── preprocess/
    └── preprocess_fdst.py          # FDST 預處理 (與 backbone 無關)
```

---

## 1. `models/vgg16bn.py` — 被替換的目標

```python
class VGG16_BN(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        vgg = models.vgg16_bn(pretrained=True)
        features = list(vgg.features.children())
        lids = [0, 33, 43]
        self.encoders = nn.ModuleList(
            nn.Sequential(*features[a:b]) for a, b in zip(lids[:-1], lids[1:])
        )
        self.num_channels = [512, 512]
        self.num_stage = len(self.num_channels)
        self.fuse_layer = UpSample_P2P(self.num_channels, ouc=256, bn=False, relu=False)

        self.decoders = SimpleDecoder(
            in_channel=self.fuse_layer.fuse_channel,
            fea_channel=self.fuse_layer.fuse_channel,
            up_scale=2, out_channel=1
        )

        self.temporal_in_channels = 1
        self.temporal_hidden = 32
        self.temporal_kernel = 3
        self.to_temporal = nn.Conv2d(self.temporal_in_channels, self.temporal_in_channels, kernel_size=1)
        self.temporal = TemporalUnit(mode='convgru', in_ch=self.temporal_in_channels,
                                     hid_ch=self.temporal_hidden, kernel_size=self.temporal_kernel)
        self.hidden2logit = nn.Conv2d(self.temporal_hidden, 1, kernel_size=1)
        self.down = 4   # ← 新模型必須設置的正確下採樣倍數

    def forward(self, image, prev_h=None, need_fp=False):
        fea2 = self.encoding(image)         # 多尺度特徵提取
        if need_fp:
            fea2 = F.dropout2d(fea2, p=0.5)
        denmap = self.decoding(fea2)         # → (B, 1, H_out, W_out)
        x_t = self.to_temporal(denmap)
        next_h = self.temporal(x_t, prev_h)  # ConvGRU
        pred_logits = self.hidden2logit(next_h)
        return pred_logits, next_h

    def encoding(self, x):
        feas = []
        for module in self.encoders:
            feas.append(x := module(x))
        feas = feas[-self.num_stage:]
        fea = self.fuse_layer(feas)
        return fea

    def decoding(self, fea2):
        denmap = self.decoders(fea2)
        if denmap.size(1) > 1:
            den1, den2 = denmap[:, :1], denmap[:, 1:2]
            den = den2 - den1
        else:
            den = denmap
        return den

    def init_hidden(self, batch_size, device=None, spatial_size=None, dtype=torch.float32):
        if device is None:
            device = next(self.parameters()).device
        if spatial_size is None:
            if hasattr(self, "_temporal_spatial_size"):
                H, W = self._temporal_spatial_size
            else:
                raise ValueError("spatial_size must be provided on first call")
        else:
            H, W = spatial_size
            self._temporal_spatial_size = (H, W)
        return torch.zeros(batch_size, self.temporal_hidden, H, W, device=device, dtype=dtype)
```

### 新模型必須實作的接口

```python
class NewBackbone(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.down = 4  # density map / 原圖 的縮放比

    def forward(self, image, prev_h=None, need_fp=False):
        """
        image:  (B, 3, H, W)
        prev_h: None 或 (B, hid_ch, H_out, W_out)
        returns:
            pred_logits: (B, 1, H_out, W_out)   — 密度圖 logits
            next_h:      (B, hid_ch, H_out, W_out) — 下一幀的 hidden state
        """
        raise NotImplementedError

    def init_hidden(self, batch_size, device, spatial_size):
        """
        spatial_size: (H_out, W_out)
        returns: (B, hid_ch, H_out, W_out)
        """
        raise NotImplementedError
```

---

## 2. `models/utils.py` — 可復用的共用組件

```python
class UpSample_P2P(nn.Module):
    """多尺度特徵融合。
    incs: list[int] — 各 encoder stage 的通道數 (如 [512, 512])
    ouc: int       — 融合後輸出通道
    forward(xs): xs 為 list[Tensor(B, C_i, H_i, W_i)]
                  返回 (B, ouc, H_out, W_out)
    """
    def __init__(self, incs, ouc, bn=True, relu=True): ...

class SimpleDecoder(nn.Sequential):
    """密度圖解碼器。
    in_channel → fea_channel → out_channel * up_scale² → PixelShuffle
    """
    def __init__(self, in_channel=128, fea_channel=64, up_scale=1, out_channel=1): ...

class ConvGRUCell(nn.Module):
    def __init__(self, in_ch, hid_ch, kernel_size=3): ...
    def forward(self, x, h):
        """x: (B, in_ch, H, W), h: (B, hid_ch, H, W) → (B, hid_ch, H, W)"""

class TemporalUnit(nn.Module):
    def __init__(self, mode='convgru', in_ch=1, hid_ch=32, kernel_size=3): ...
    def forward(self, x, h):
        """x: (B, in_ch, H, W), h: (B, hid_ch, H, W) or None → (B, hid_ch, H, W)"""
```

---

## 3. `models/__init__.py` — 模型替換處

```python
from .vgg16bn import VGG16_BN
def build_model(config):
    model = {
        'vgg16bn': VGG16_BN       # ← 加入新模型映射
    }[config.NAME.lower()]

    return model(config), model(config)  # returns (student, teacher)
```

### 新模型接入方式

```python
from .resnet import ResNet50     # 新的 backbone
def build_model(config):
    model = {
        'vgg16bn': VGG16_BN,
        'resnet50': ResNet50,     # ← 加入
    }[config.NAME.lower()]
    return model(config), model(config)
```

啟動指令用 `--opts MODEL.NAME resnet50`。

---

## 4. `config.py` — 配置相關字段

```python
_C.MODEL = CN()
_C.MODEL.NAME = 'VGG16BN'      # 字串映射到 models/__init__.py
_C.MODEL.RESUME = ''            # resume checkpoint 路徑
_C.MODEL.FACTOR = 1             # 傳給 loss (可忽略)
_C.MODEL.LOSS = 'P2R'           # 損失函數選擇

_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 50
_C.TRAIN.BASE_LR = 5e-5         # 非 backbone 參數的 LR
_C.TRAIN.BACKBONE_LR = 1e-5     # backbone 參數的 LR (在編碼器部分)
```

---

## 5. `main.py` — 訓練入口（關鍵調用點）

```python
# === 模型建立 ===
def main_worker(config):
    student, teacher = build_model(config.MODEL)        # ← 工廠模式
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad = False
    student.cuda(); teacher.cuda()

    # === 損失函數 ===
    criterion, test_criterion = build_loss(config.MODEL)

    # === 優化器（backbone 與非 backbone 分組不同 LR） ===
    param_dicts = [
        {
            "params": [p for n, p in student.named_parameters()
                       if "encoders" not in n and p.requires_grad]
        }, {
            "params": [p for n, p in student.named_parameters()
                       if "encoders" in n and p.requires_grad],
            "lr": config.TRAIN.BACKBONE_LR,
        },
    ]
    optimizer = optim.Adam(param_dicts, lr=config.TRAIN.BASE_LR, ...)

    # === 訓練循環 ===
    for epoch in ...:
        temporal_train_one_epoch(
            epoch=epoch,
            dataloader=data_loader_train,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            device=torch.device('cuda'),
            down=getattr(student, 'down', 4),   # ← 新模型需有 self.down
            ema_alpha=0.999,
            scaler=scaler,
            p2r_loss_cfg=None,
            amp_enabled=True,
            min_batch_size=2,
            stage=stage,
            stage1_epochs=STAGE_1,
            total_epochs=config.TRAIN.EPOCHS,
        )
```

### 優化器分組注意

如果新 backbone 編碼器參數名稱不含 `encoders`，需修改優化器分組邏輯：

```python
# 替代方案：用名稱前綴匹配
backbone_prefixes = ('layer1', 'layer2', 'layer3', 'layer4')  # ResNet 範例
param_dicts = [
    {
        "params": [p for n, p in student.named_parameters()
                   if not n.startswith(backbone_prefixes) and p.requires_grad]
    }, {
        "params": [p for n, p in student.named_parameters()
                   if n.startswith(backbone_prefixes) and p.requires_grad],
        "lr": config.TRAIN.BACKBONE_LR,
    },
]
```

---

## 6. `train/train_loop.py` — 訓練迴圈（模型調用方式）

```python
def train_one_epoch(epoch, dataloader, student, teacher, optimizer, device,
                    down=4, ema_alpha=0.999, scaler=None, p2r_loss_cfg=None,
                    amp_enabled=True, min_batch_size=2, stage='sup',
                    stage1_epochs=25, total_epochs=50):
    student.train()
    teacher.eval()
    p2r = P2RLoss()

    for batch in dataloader:
        limg_batch, lseqs, lids, uimg_batch, umask_batch, uids, uflows_batch = batch

        # === 監督損失（第一幀） ===
        lframe = limg_batch[:, 0]
        lpred, _ = student(lframe, prev_h=None)    # ← model.forward(img, prev_h)
        ldots = [seq[0][0][:, :2].to(device) for seq in lseqs]
        sup_loss = p2r(lpred, ldots, down=down)

        if stage == 'sup':
            # 僅監督：backward + optimizer.step()

        elif stage == 'semi':
            # === 半監督（BPTT across frames） ===
            student_h = student.init_hidden(B, spatial_size=(H//down, W//down))
            teacher_h = teacher.init_hidden(B, spatial_size=(H//down, W//down))

            for t in range(1, T):
                student_pred_t, student_h = student(frame_t, student_h)
                # 用 teacher pseudo-label + flow warp 做目標
                points_prev_list = generate_points_from_density(teacher_pred_prev, threshold=0.3)
                warped_points_list = warp_points_with_flow(points_prev_list, flow_t_minus, ...)
                seqs_for_loss = prepare_p2r_seqs_from_points_list(warped_points_list)
                loss_p2r = p2r(student_pred_t, seqs_for_loss, down=down)

                loss = loss_p2r + sup_loss / (T - 1)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                student_h = student_h.detach()
                update_ema(student, teacher, ema_alpha)   # teacher = α*t + (1-α)*s
```

---

## 7. `losses/p2rloss.py` — 損失函數（與 backbone 無關）

```python
class P2RLoss(nn.modules.loss._Loss):
    def forward(self, dens, seqs, down, masks=None, crop_den_masks=None, pos_weight=None):
        """
        dens:  (B, 1, H, W) — student 輸出的密度 logits
        seqs:  list[B] of (N_i, 2) — GT 點座標 (x, y) 在原始圖像解析度
        down:  密度圖 / 原圖 的縮放比 (即 self.down)
        pos_weight: 可選，動態正樣本權重
        returns: scalar loss
        """
```

**注意**: `down` 參數在 loss 內用於將像素座標映射回原圖空間。新模型的 `self.down` 必須與此一致。

---

## 8. `datasets/fdst.py` — FDST 資料集（與 backbone 無關）

```python
class FDST(data.Dataset):
    def __getitem__(self, index):
        # 訓練時返回 7 元素:
        #   (limg_seq, lseqs, lid, uimg_seq, umask_seq, uid, uflows)
        # 推理時返回 3~4 元素:
        #   (limg_seq, lseqs, lid, [flows])
        pass

    @staticmethod
    def collate_fn(samples):
        # 打包成 batch:
        #   limg_batch: (B, T, C, H, W)
        #   lseqs: list of list of tensor
        #   uflows_batch: (B, T-1, 2, H_flow, W_flow) or None
```

---

## 9. `utils.py` — Checkpoint 格式

```python
def save_checkpoint(config, epoch, model, optimizer, lr_scheduler, scaler, max_accuracy, logger):
    teacher, student = model
    save_state = {
        'teacher': teacher.state_dict(),
        'student': student.state_dict(),
        'epoch': epoch,
        'optimizer': optimizer.state_dict() if optimizer else None,
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
        'scaler': scaler.state_dict() if scaler else None,
        'max_accuracy': max_accuracy,
    }

def load_checkpoint(config, model, optimizer, lr_scheduler, scaler, logger):
    teacher, student = model
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    teacher.load_state_dict(checkpoint['teacher'], strict=False)
    student.load_state_dict(checkpoint['student'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])
    return saved_epoch, max_accuracy
```

**注意**: Checkpoint 儲存 `teacher` + `student` 兩組權重。新模型的 `state_dict()` key 名稱必須與 checkpoint 相容，否則 resume 會失敗。

---

## 10. 替換 VGG16 為新模型的檢查清單

| 步驟 | 檔案 | 需確認事項 |
|------|------|-----------|
| 1 | `models/new_backbone.py` | 實作 `forward(image, prev_h)` → `(logits, next_h)` |
| 2 | `models/new_backbone.py` | 實作 `init_hidden(batch_size, device, spatial_size)` |
| 3 | `models/new_backbone.py` | 設置 `self.down = N` 與 loss 中的 `down` 參數一致 |
| 4 | `models/__init__.py` | 加入 import + dict 映射 |
| 5 | `main.py` | 若新 backbone 參數名稱不含 `encoders`，修改優化器分組條件 |
| 6 | `models/new_backbone.py` | 可選用 `UpSample_P2P`、`SimpleDecoder`、`TemporalUnit` 等共用組件 |
| 7 | 訓練指令 | `--opts MODEL.NAME new_backbone` |

### VGG16 與常見 backbone 對照

| 特性 | VGG16 | ResNet50 | HRNet-W32 | MobileNetV3 |
|------|-------|----------|-----------|-------------|
| 參數量 | ~138M (全網) | ~25M | ~28M | ~5M |
| 多尺度特徵 | 2 級 (512, 512) | 4 級 (256,512,1024,2048) | 4 解析度並行 | 3 級 |
| 典型 `self.down` | 4 | 4 或 8 | 4 | 4 |
| 預訓練 | ImageNet | ImageNet | ImageNet | ImageNet |
| 修改量 | 基準 | 中 (須處理多尺度聚合) | 高 (並行分支) | 低 (輕量) |
