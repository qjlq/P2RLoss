# P2RLoss

Official code for CVPR-2025 paper: "[Point-to-Region Loss for Semi-Supervised Point-Based Crowd Counting](https://arxiv.org/abs/2505.21943)."

## Overview

This repository implements the P2RLoss method for semi-supervised point-based crowd counting. The project uses a teacher-student framework with VGG16BN backbone and ConvGRU temporal modeling, and proposes a novel Point-to-Region Loss to improve counting performance with limited labeled data.

## Features

- **Semi-supervised learning**: Utilizes both labeled and unlabeled data for training
- **Teacher-student framework**: EMA-based teacher model for pseudo-labeling
- **P2RLoss**: Novel loss that matches points to regions for better supervision
- **Temporal modeling**: ConvGRU-based temporal unit for video sequence processing
- **Flow-based warping**: Optical flow warps teacher pseudo-points across frames for temporal consistency
- **Truncated BPTT**: Backpropagation through time with truncated sequences
- **Mixed precision (AMP)**: Automatic mixed precision training support
- **Multi-dataset**: Supports ShanghaiTech (SHHA) and FDST video datasets
- **Strong augmentation**: Color jitter, grayscale, Gaussian blur for unlabeled samples

## Requirements

- Python 3.x
- PyTorch >= 1.8
- torchvision
- timm
- yacs
- numpy
- PIL
- opencv-python
- tqdm
- matplotlib
- termcolor

## Installation

```bash
# Clone the repository
git clone https://github.com/Elin24/P2RLoss.git
cd P2RLoss

# Install dependencies
pip install torch torchvision timm yacs numpy opencv-python tqdm matplotlib termcolor
```

## Data Preparation

### ShanghaiTech (SHHA)

```
data/
└── ShanghaiTech/
    └── part_A/
        ├── train_data/
        │   ├── images/            # .jpg frames
        │   └── new-anno/          # GT_{id}.npy point annotations
        └── test_data/
            ├── images/
            └── new-anno/
```

### FDST (Video Dataset)

```
data/
└── FDST/
    ├── train_data/
    │   └── images/
    │       ├── video_001/         # per-video subdirectories
    │       │   ├── frame_0001.jpg
    │       │   └── ...
    │       └── video_002/
    └── test_data/
        └── images/
```

Annotation files (`.npy`) are searched under `new-anno/` or `annotations/` directories with patterns: `GT_{video}_{frame}.npy` or `GT_{frame}.npy`.

### Protocol File

You need a protocol file listing labeled image IDs (one per line) for semi-supervised splitting.

### Optical Flow (Optional)

Precomputed flow files can be placed under a `flow_root` directory with naming convention `{frame}_flow.npy` or `{video}_{frame}_flow.npy`. Flow shape: `(T-1, 2, H, W)` per sequence.

## Training

### Basic Usage

```bash
python main.py --data-path /path/to/data \
    --label 5 \
    --protocol /path/to/protocol.txt \
    --batch-size 16 \
    --tag experiment_name
```

`--label` specifies the percentage of labeled data (e.g., `5` = 5%).

### Using the provided script

Modify `run.sh` with your data paths:

```bash
datadir=/path/to/ShanghaiTech/part_A
name=sha
part=5
T=${name}-L${part}

mkdir exp
mkdir exp/$T
mkdir exp/$T/code
cp -r datasets exp/$T/code/datasets
cp -r models exp/$T/code/models
cp -r losses exp/$T/code/losses
cp ./*.py exp/$T/code/
cp run.sh exp/$T/code

mkdir exp/$T/train.log
python main.py --data-path $datadir \
    --label ${part} --protocol /path/to/protocol.txt \
    --batch-size 16 --tag $T 2>&1 | tee exp/$T/train.log/running.log
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--data-path` | Path to the dataset |
| `--label` | Percentage of labeled data (e.g., 5 for 5%) |
| `--protocol` | Path to the protocol file listing labeled IDs |
| `--batch-size` | Batch size for training |
| `--tag` | Experiment tag for logging |
| `--resume` | Path to checkpoint for resuming training |
| `--eval` | Evaluation mode only |
| `--use-checkpoint` | Enable gradient checkpointing to save memory |
| `--accumulation-steps` | Gradient accumulation steps |
| `--output` | Root output folder (default: `exp/<tag>/output`) |

## Configuration

The training configuration is managed via `config.py` using yacs `CfgNode`. Key parameters:

- `DATA.BATCH_SIZE`: Batch size (default: 1)
- `DATA.DATASET`: Dataset name — `'shha'` or `'fdst'` (default: `'shha'`)
- `MODEL.NAME`: Model architecture (default: `'VGG16BN'`)
- `MODEL.LOSS`: Loss function (default: `'P2R'`)
- `TRAIN.EPOCHS`: Total training epochs (default: 1500)
- `TRAIN.BASE_LR`: Base learning rate (default: 5e-5)
- `TRAIN.BACKBONE_LR`: Backbone learning rate (default: 1e-5)
- `TRAIN.LR_SCHEDULER.DECAY_EPOCHS`: LR decay interval (default: 3500)
- `TRAIN.LR_SCHEDULER.DECAY_RATE`: LR decay rate (default: 0.9)

## Code Structure

```
P2RLoss/
├── config.py              # Configuration management (yacs CfgNode)
├── main.py                # Main training and evaluation script
├── run.sh                 # Training shell script
├── utils.py               # Utility functions (checkpointing, plotting, seeding)
├── logger.py              # Logging utilities (colored console + file)
├── lr_scheduler.py        # Learning rate schedulers (cosine, step, linear)
├── datasets/              # Dataset loaders
│   ├── __init__.py        # Dataset builder with DataLoader
│   ├── shha.py            # ShanghaiTech dataset (sequence + flow support)
│   ├── fdst.py            # FDST video dataset (sequence + flow support)
│   └── utils.py           # NormalSample, augmentations, cropping
├── models/                # Model architectures
│   ├── __init__.py        # Model builder
│   ├── vgg16bn.py         # VGG16BN backbone + ConvGRU temporal unit
│   └── utils.py           # UpSample_P2P, SimpleDecoder, ConvGRU, TemporalUnit
├── losses/                # Loss functions
│   ├── __init__.py
│   └── p2rloss.py         # Point-to-Region Loss (P2RLoss)
└── train/
    └── train_loop.py      # Training loop with truncated BPTT, EMA, AMP, flow warping
```

## Model Architecture

The model uses VGG16BN backbone with:
- **Multi-scale feature extraction**: Encoders from VGG16BN stages
- **Feature fusion**: `UpSample_P2P` aligns and fuses multi-scale features
- **Decoder**: `SimpleDecoder` with PixelShuffle upscaling to produce density maps
- **Temporal unit**: ConvGRU cell for temporal modeling across video frames
- **Output projection**: `hidden2logit` maps ConvGRU hidden state to logits (1 channel)

The model supports temporal forward passes: `forward(image, prev_h=None) -> (pred_logits, next_h)`.

## Training Strategy

1. **Stage 1 (Epochs 0-24)**: Supervised training with labeled data only
2. **Stage 2 (Epochs 25-49)**: Semi-supervised training with both labeled and unlabeled data
   - Teacher model generates pseudo-points from previous frame predictions
   - Pseudo-points are warped to current frame via optical flow (when available)
   - Student model learns from P2R loss between its predictions and warped pseudo-points
   - Teacher weights updated via EMA: `teacher = α·teacher + (1-α)·student`
   - Truncated BPTT: hidden states detached after each time step
3. **Post Stage 2 (Epochs 50-1499)**: Continued training with saved stage-2 checkpoint

## Temporal Training Details

- Dataset returns sequences of frames: shape `(T, C, H, W)` per sample
- Training loop iterates over time steps `t = 1..T-1`:
  - Student forward on frame `t`
  - Teacher pseudo-points from frame `t-1` warped by flow `t-1→t`
  - P2R loss computed between student prediction and warped pseudo-points
  - Teacher produces new prediction for frame `t` (used as next `t-1` reference)
- Flow files are optional; without flow, pseudo-points are used directly

## Datasets

### ShanghaiTech (SHHA)
- Standard crowd counting benchmark
- Training returns 7-tuple: `(labeled_imgs, labeled_dots, labeled_ids, unlabeled_imgs, unlabeled_masks, unlabeled_ids, flows)`
- Random crop + resize augmentation
- Random horizontal flip

### FDST
- Video crowd counting dataset with per-video subdirectories
- Same sequence/flow API as SHHA
- Supports multiple annotation directory layouts

## Results

The model achieves competitive performance on ShanghaiTech dataset with limited labeled data. Training logs and curves (MAE, MSE, loss) are saved to `exp/<tag>/train.log/`.

## Citation

If you find this code useful for your research, please cite:

```bibtex
@inproceedings{lin2025point,
  title={Point-to-Region Loss for Semi-Supervised Point-Based Crowd Counting},
  author={Lin, Wei and Zhao, Chenyang and Chan, Antoni B},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={29363--29373},
  year={2025}
}
```
