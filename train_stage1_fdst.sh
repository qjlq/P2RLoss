#!/bin/bash
# ============================================================================
# FDST Stage 1 監督預訓練腳本
# 兩種模式: VGG16BN (200 epochs) + EMAC (100 epochs)
# 使用 --sup-only，永遠不進 semi 階段
# AUTO_RESUME=True，中斷後重跑會自動從最新 checkpoint 繼續
# ============================================================================

set -e

DATA="/media/SSD/OSshareSpace/localSpace/workPlace/zhongKe/gd/FDST"
PROTOCOL="/media/SSD/OSshareSpace/localSpace/workPlace/zhongKe/gd/FDST/FDST_protocol.txt"
PROJECT_DIR="/media/SSD/OSshareSpace/localSpace/workPlace/zhongKe/gd/myP2R/P2RLoss"

cd "$PROJECT_DIR"

echo "============================================================"
echo " FDST Stage 1 監督預訓練"
echo " 模式 1: VGG16BN  200 epochs"
echo " 模式 2: EMAC     100 epochs"
echo " 使用 --sup-only, AUTO_RESUME=True"
echo "============================================================"
echo ""

# ══════════════════════════════════════════════════════════════════
# Mode 1: VGG16BN — 200 epochs
# ══════════════════════════════════════════════════════════════════
echo "────────────────────────────────────────────────────────────"
echo " 🔄 Mode 1: VGG16BN 監督訓練 (200 epochs)"
echo "    開始時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "────────────────────────────────────────────────────────────"

python main.py \
    --data-path $DATA \
    --label 5 --protocol $PROTOCOL \
    --batch-size 32 --tag fdst-s1-vgg16bn \
    --sup-only \
    --opts DATA.DATASET fdst \
        DATA.NUM_WORKERS 16 \
        TRAIN.EPOCHS 200 TRAIN.AUTO_RESUME True

echo "✅ Mode 1 完成"
echo ""

# ══════════════════════════════════════════════════════════════════
# Mode 2: EMAC — 100 epochs
# ══════════════════════════════════════════════════════════════════
echo "────────────────────────────────────────────────────────────"
echo " 🔄 Mode 2: EMAC 監督訓練 (100 epochs)"
echo "    開始時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "────────────────────────────────────────────────────────────"

python main.py \
    --data-path $DATA \
    --label 5 --protocol $PROTOCOL \
    --batch-size 16 --tag fdst-s1-emac \
    --sup-only \
    --opts DATA.DATASET fdst MODEL.NAME emac \
        DATA.SEQ_LEN 2 DATA.SEQ_STRIDE 1 \
        DATA.NUM_WORKERS 16 \
        TRAIN.EPOCHS 100 TRAIN.AUTO_RESUME True

echo "✅ Mode 2 完成"
echo ""
echo "============================================================"
echo " 🎯 全部完成！"
echo "    結束時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
