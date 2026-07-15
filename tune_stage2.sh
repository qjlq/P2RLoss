#!/bin/bash
# ============================================================================
# EMAC Stage 2 超參數調優腳本
# 目標：定位 MAE 在 Stage 2 從 11→20 持續無法下降的根因
# 每組實驗從 epoch 25 resume，訓練至 epoch 74 (共 50 epochs)
# 預估耗時: ~4h/run × 4 runs = ~16h
#
# 錯誤處理: 單一實驗失敗不影響後續實驗，顯示 FAIL 但繼續執行
# ============================================================================

# 不使用 set -e，改用手動錯誤檢查以確保所有實驗可繼續
set -o pipefail

# ── 路徑設定 ───────────────────────────────────────────────────────────────
PROJECT_DIR="/media/SSD/OSshareSpace/localSpace/workPlace/zhongKe/gd/myP2R/P2RLoss"
CKPT="${PROJECT_DIR}/exp/fdst-down2_emac_20260710_185758/output/ckpt_epoch_best_epoch25.pth"
DATA="/media/SSD/OSshareSpace/localSpace/workPlace/zhongKe/gd/FDST"
PROTOCOL="/media/SSD/OSshareSpace/localSpace/workPlace/zhongKe/gd/FDST/FDST_protocol.txt"
TRAIN_LOOP="${PROJECT_DIR}/train/train_loop.py"

cd "$PROJECT_DIR"

# ── 確保中斷或失敗時恢復原始檔案 ───────────────────────────────────────────
cleanup() {
    echo ""
    echo "⚠️  腳本被中斷，正在恢復原始 train_loop.py..."
    if [ -f "${TRAIN_LOOP}.bak" ]; then
        cp "${TRAIN_LOOP}.bak" "$TRAIN_LOOP"
        rm -f "${TRAIN_LOOP}.bak"
        echo "✅ train_loop.py 已恢復"
    fi
    echo "⚠️  腳本退出"
}
trap cleanup EXIT SIGINT SIGTERM

# ── 共用命令前綴 ───────────────────────────────────────────────────────────
BASE_CMD="python main.py \
    --data-path $DATA --label 5 --protocol $PROTOCOL --batch-size 16 \
    --resume $CKPT \
    --opts DATA.DATASET fdst MODEL.NAME emac \
        DATA.SEQ_LEN 2 DATA.SEQ_STRIDE 1 DATA.NUM_WORKERS 16 \
        TRAIN.EPOCHS 75 TRAIN.AUTO_RESUME False"

# ── 備份原始 train_loop.py ────────────────────────────────────────────────
cp "$TRAIN_LOOP" "${TRAIN_LOOP}.bak"

# ── 輔助函數 ──────────────────────────────────────────────────────────────

set_pseudo_thresh() {
    sed -i "s/pseudo_thresh = [0-9.]\+ if stage == 'semi' else 0.3/pseudo_thresh = $1 if stage == 'semi' else 0.3/g" "$TRAIN_LOOP"
}

set_stage2_pos_weight() {
    sed -i "s/return [0-9.]\+ \* (1.0 - alpha) + [0-9.]\+ \* alpha/return $1 \* (1.0 - alpha) + $2 \* alpha/" "$TRAIN_LOOP"
    sed -i "s/^        return 2\.0$/        return $2/" "$TRAIN_LOOP"
}

set_noise() {
    sed -i "s/torch\.randn_like(uimg_batch) \* [0-9.]\+/torch.randn_like(uimg_batch) * $1/g" "$TRAIN_LOOP"
}

set_opt_weight() {
    sed -i "s/opt_loss = F\.mse_loss(img_warp, cur_rgb) \* [0-9.]\+/opt_loss = F.mse_loss(img_warp, cur_rgb) * $1/g" "$TRAIN_LOOP"
}

set_tv_weight() {
    sed -i "s/flow_tv = tv_loss(flo_t) \* [0-9.]\+/flow_tv = tv_loss(flo_t) * $1/g" "$TRAIN_LOOP"
}

run_experiment() {
    local exp_num=$1
    local exp_name=$2
    local cmd=$3

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo " 🔄 [$exp_num/4] $exp_name"
    echo "     開始時間: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "     命令: $cmd"
    echo "────────────────────────────────────────────────────────────"
    echo ""

    # 執行實驗，捕獲退出碼但不終止腳本
    eval "$cmd"
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "✅ [$exp_num/4] $exp_name 完成"
    else
        echo "❌ [$exp_num/4] $exp_name 失敗 (exit code=$exit_code)"
    fi

    return $exit_code
}

# ── 驗證 sed 模式匹配 ───────────────────────────────────────────────────
echo "🔍 驗證 sed 模式匹配..."
grep -n "pseudo_thresh = " "$TRAIN_LOOP"
grep -n "return.*\* (1.0 - alpha) + .* \* alpha" "$TRAIN_LOOP"
grep -n "^        return [0-9.]\+$" "$TRAIN_LOOP"
grep -n "randn_like(uimg_batch) \* " "$TRAIN_LOOP"
grep -n "opt_loss = F.mse_loss" "$TRAIN_LOOP"
grep -n "flow_tv = tv_loss" "$TRAIN_LOOP"

# 確認所有模式都匹配
PATTERNS_OK=true
grep -q "pseudo_thresh = 0.85 if stage == 'semi' else 0.3" "$TRAIN_LOOP" || { echo "❌ pseudo_thresh 模式不匹配"; PATTERNS_OK=false; }
grep -q "return 20.0 \* (1.0 - alpha) + 2.0 \* alpha" "$TRAIN_LOOP" || { echo "❌ pos_weight decay 模式不匹配"; PATTERNS_OK=false; }
grep -q "noise = torch.randn_like(uimg_batch) \* 0.05" "$TRAIN_LOOP" || { echo "❌ noise 模式不匹配"; PATTERNS_OK=false; }

if [ "$PATTERNS_OK" = false ]; then
    echo ""
    echo "❌ sed 模式匹配失敗，請檢查 train/train_loop.py 是否被意外修改"
    cleanup
    exit 1
fi
echo "✅ 模式驗證完成"
echo ""

echo "============================================================"
echo " EMAC Stage 2 調優腳本開始"
echo " 恢復 checkpoint: epoch 25"
echo " 每組 50 epochs (25→74), 4 組 × ~4h = ~16h"
echo " 每個實驗獨立執行，失敗不影響後續"
echo "============================================================"

# ── 紀錄實驗結果 ──────────────────────────────────────────────────────────
RESULTS=()

# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Baseline (對照組)
#   pseudo_thresh = 0.85, pos_weight = 20→2, noise = 0.05
# ═══════════════════════════════════════════════════════════════════════════
set_pseudo_thresh 0.85
set_stage2_pos_weight 20.0 2.0
set_noise 0.05
set_opt_weight 0.05
set_tv_weight 0.01

if run_experiment 1 "Baseline (預設配置)" "$BASE_CMD --tag emac_sweep1_baseline"; then
    RESULTS+=("✅ 1 baseline: 成功")
else
    RESULTS+=("❌ 1 baseline: 失敗")
fi

# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: 高精度偽標籤 + 強光流監督
#   pseudo_thresh = 0.95, opt_loss × 5 = 0.25
# ═══════════════════════════════════════════════════════════════════════════
set_pseudo_thresh 0.95
set_stage2_pos_weight 20.0 2.0
set_noise 0.05
set_opt_weight 0.25
set_tv_weight 0.01

if run_experiment 2 "高精度偽標籤 (thresh=0.95) + 強光流 (opt×5)" \
    "$BASE_CMD --tag emac_sweep2_highqual"; then
    RESULTS+=("✅ 2 highqual: 成功")
else
    RESULTS+=("❌ 2 highqual: 失敗")
fi

# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: 保守更新 — 低 LR + 固定低 pos_weight
#   BASE_LR = 2e-5, pos_weight = 5→5 (fixed)
# ═══════════════════════════════════════════════════════════════════════════
set_pseudo_thresh 0.85
set_stage2_pos_weight 5.0 5.0
set_noise 0.05
set_opt_weight 0.05
set_tv_weight 0.01

if run_experiment 3 "保守更新 (LR=2e-5, pos=5→5)" \
    "$BASE_CMD TRAIN.BASE_LR 2e-5 --tag emac_sweep3_lowlr"; then
    RESULTS+=("✅ 3 lowlr: 成功")
else
    RESULTS+=("❌ 3 lowlr: 失敗")
fi

# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: 強噪聲 + 強正則
#   noise = 0.10, drop_path = 0.4, pos_weight = 3→3
# ═══════════════════════════════════════════════════════════════════════════
set_pseudo_thresh 0.85
set_stage2_pos_weight 3.0 3.0
set_noise 0.10
set_opt_weight 0.05
set_tv_weight 0.01

if run_experiment 4 "強噪聲 (noise=0.10) + 強正則 (drop=0.4)" \
    "$BASE_CMD EMAC_DROP_PATH 0.4 --tag emac_sweep4_strongreg"; then
    RESULTS+=("✅ 4 strongreg: 成功")
else
    RESULTS+=("❌ 4 strongreg: 失敗")
fi

# ── 還原原始檔案 ───────────────────────────────────────────────────────────
cp "${TRAIN_LOOP}.bak" "$TRAIN_LOOP"
rm -f "${TRAIN_LOOP}.bak"

# ── 打印最終結果 ───────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " 🎯 調優實驗結果"
echo "============================================================"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
echo " 結果目錄:"
echo "   exp/emac_sweep1_baseline_<timestamp>/output/"
echo "   exp/emac_sweep2_highqual_<timestamp>/output/"
echo "   exp/emac_sweep3_lowlr_<timestamp>/output/"
echo "   exp/emac_sweep4_strongreg_<timestamp>/output/"
echo ""
echo " 對比 MAE 曲線:"
echo "   grep \"^* MAE\" exp/emac_sweep1_*/output/log_rank0.txt"
echo "============================================================"
echo " 結束時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
