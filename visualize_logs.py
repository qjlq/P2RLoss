#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = '.'

def parse_stage1_losses(filepath):
    vgg_epochs, vgg_losses = [], []
    mode = None
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if 'VGG16BN' in line and '監督訓練' in line:
                mode = 'VGG16BN'
            elif 'EMAC' in line and '監督訓練' in line:
                mode = 'EMAC'
            m = re.search(r'Epoch \[(\d+)\] avg_loss: ([\d.eE+-]+)', line)
            if m and mode == 'VGG16BN':
                vgg_epochs.append(int(m.group(1)))
                vgg_losses.append(float(m.group(2)))
    return vgg_epochs, vgg_losses


def parse_stage1_test_metrics(filepath):
    vgg_epochs, vgg_mae, vgg_mse = [], [], []
    emac_epochs, emac_mae, emac_mse = [], [], []
    mode = None
    cur_epoch = 0
    emac_idx = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if 'VGG16BN' in line and '監督訓練' in line:
                mode = 'VGG16BN'
                cur_epoch = 0
                emac_idx = 0
            elif 'EMAC' in line and '監督訓練' in line:
                mode = 'EMAC'
            m = re.search(r'Epoch \[(\d+)\] avg_loss:', line)
            if m:
                cur_epoch = int(m.group(1))
            m = re.search(r'\* MAE ([\d.]+) MSE ([\d.]+)', line)
            if m:
                mae, mse = float(m.group(1)), float(m.group(2))
                if mode == 'VGG16BN':
                    vgg_epochs.append(cur_epoch)
                    vgg_mae.append(mae)
                    vgg_mse.append(mse)
                elif mode == 'EMAC':
                    emac_epochs.append(emac_idx)
                    emac_idx += 1
                    emac_mae.append(mae)
                    emac_mse.append(mse)
    return (vgg_epochs, vgg_mae, vgg_mse), (emac_epochs, emac_mae, emac_mse)


def parse_tune2_metrics(filepath):
    experiments = []
    current_exp, current_maes, current_mses = None, [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.search(r'\[(\d)/4\] (.+?)(?: 完成|$)', line)
            if m and ('🔄' in line or '✅' in line):
                if current_exp is not None and current_maes:
                    experiments.append((current_exp, current_maes, current_mses))
                current_exp = m.group(2).strip()
                current_maes, current_mses = [], []
            m = re.search(r'\* MAE ([\d.]+) MSE ([\d.]+)', line)
            if m:
                current_maes.append(float(m.group(1)))
                current_mses.append(float(m.group(2)))
    if current_exp is not None and current_maes:
        experiments.append((current_exp, current_maes, current_mses))
    return experiments


def plot_stage1_loss(vgg_epochs, vgg_losses, savepath):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(vgg_epochs, vgg_losses, 'b-', linewidth=1.5, label='VGG16BN')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Avg Loss')
    ax.set_title('Stage 1 Training Loss (VGG16BN)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    best_idx = np.argmin(vgg_losses)
    ax.plot(vgg_epochs[best_idx], vgg_losses[best_idx], 'ro', markersize=6)
    ax.annotate(f'Best: {vgg_losses[best_idx]:.6f} @ epoch {vgg_epochs[best_idx]}',
                xy=(vgg_epochs[best_idx], vgg_losses[best_idx]),
                xytext=(10, -15), textcoords='offset points', fontsize=9, color='red')
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {savepath}')


def plot_stage1_test(vgg_data, emac_data, savepath):
    (ev, mae_v, mse_v), (ee, mae_e, mse_e) = vgg_data, emac_data
    has_vgg = len(ev) > 0
    has_emac = len(ee) > 0
    n = (has_vgg + has_emac)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(8 * max(n, 1), 5))

    if n == 1:
        axes = [axes]

    idx = 0
    if has_vgg:
        ax = axes[idx]; idx += 1
        ax.plot(ev, mae_v, 'r-', label='MAE', linewidth=1)
        ax.plot(ev, mse_v, 'b-', label='MSE', linewidth=1)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Metric')
        ax.set_title('VGG16BN Test MAE/MSE')
        ax.legend(); ax.grid(True, alpha=0.3)
        best_m = np.argmin(mae_v); best_s = np.argmin(mse_v)
        ax.plot(ev[best_m], mae_v[best_m], 'ro', markersize=5)
        ax.plot(ev[best_s], mse_v[best_s], 'bo', markersize=5)

    if has_emac:
        ax = axes[idx]
        ax.plot(ee, mae_e, 'r-o', label='MAE', linewidth=1, markersize=3)
        ax.plot(ee, mse_e, 'b-o', label='MSE', linewidth=1, markersize=3)
        ax.set_xlabel('Evaluation #'); ax.set_ylabel('Metric')
        ax.set_title('EMAC Test MAE/MSE')
        ax.legend(); ax.grid(True, alpha=0.3)
        best_m = np.argmin(mae_e); best_s = np.argmin(mse_e)
        ax.plot(ee[best_m], mae_e[best_m], 'ro', markersize=5)
        ax.plot(ee[best_s], mse_e[best_s], 'bo', markersize=5)

    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {savepath}')


def plot_stage1_combined(vgg_data, emac_data, savepath):
    (ev, mae_v, mse_v), (ee, mae_e, mse_e) = vgg_data, emac_data
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Epoch (VGG16BN) / Eval # (EMAC)')
    ax1.set_ylabel('MAE', color='tab:red')
    ax1.plot(ev, mae_v, 'r-', label='VGG16BN MAE', linewidth=1.5)
    if len(ee) > 0:
        offset = ev[-1] + 5 if ev else 0
        ee_shift = [e + offset for e in ee]
        ax1.plot(ee_shift, mae_e, 'r--o', label='EMAC MAE', linewidth=1.5, markersize=4)
        ax1.axvline(x=offset - 2.5, color='gray', linestyle=':', alpha=0.5)
        ax1.text(offset - 2, ax1.get_ylim()[1] * 0.95, 'VGG16BN → EMAC', fontsize=8, rotation=90)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel('MSE', color='tab:blue')
    ax2.plot(ev, mse_v, 'b-', label='VGG16BN MSE', linewidth=1.5)
    if len(ee) > 0:
        offset = ev[-1] + 5 if ev else 0
        ee_shift = [e + offset for e in ee]
        ax2.plot(ee_shift, mse_e, 'b--o', label='EMAC MSE', linewidth=1.5, markersize=4)
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1.set_title('Stage 1 Test MAE/MSE (VGG16BN → EMAC)')
    fig.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {savepath}')


def plot_tune_comparison(experiments, savepath):
    if not experiments:
        return
    names = [e[0][:25] for e in experiments]
    best_maes = [min(e[1]) for e in experiments]
    best_mses = [min(e[2]) for e in experiments]
    final_maes = [e[1][-1] for e in experiments]
    final_mses = [e[2][-1] for e in experiments]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, vals, title, color in [
        (axes[0], [(b, f) for b, f in zip(best_maes, final_maes)], 'MAE', 'coral'),
        (axes[1], [(b, f) for b, f in zip(best_mses, final_mses)], 'MSE', 'steelblue')
    ]:
        x = np.arange(len(names))
        w = 0.3
        ax.bar(x - w/2, [v[0] for v in vals], w, label=f'Best {title}', color=color, alpha=0.8)
        ax.bar(x + w/2, [v[1] for v in vals], w, label=f'Final {title}', color=color, alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right', fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(f'{title}: Best vs Final')
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i - w/2, v[0] + 0.3, f'{v[0]:.1f}', ha='center', va='bottom', fontsize=7)
            ax.text(i + w/2, v[1] + 0.3, f'{v[1]:.1f}', ha='center', va='bottom', fontsize=7)

    plt.suptitle('Tune Stage 2 Experiment Comparison', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {savepath}')


def plot_tune_series(experiments, savepath):
    if not experiments:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for ax, metric_fn, label in [
        (axes[0], lambda e: e[1], 'MAE'),
        (axes[1], lambda e: e[2], 'MSE')
    ]:
        for i, (name, maes, mses) in enumerate(experiments):
            data = metric_fn((name, maes, mses))
            ax.plot(range(len(data)), data, f'-o', color=colors[i % len(colors)],
                    label=name[:20], linewidth=1.2, markersize=3)
        ax.set_xlabel('Evaluation Step')
        ax.set_ylabel(label)
        ax.set_title(f'{label} Progression')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {savepath}')


def main():
    print('=== Parsing train_stage1_fdst_clean.log ===')
    vgg_loss = parse_stage1_losses('train_stage1_fdst_clean.log')
    vgg_test, emac_test = parse_stage1_test_metrics('train_stage1_fdst_clean.log')
    print(f'  VGG16BN: {len(vgg_loss[0])} loss, {len(vgg_test[0])} test')
    print(f'  EMAC: {len(emac_test[0])} test metrics')

    print('\n=== Parsing tune_stage2_clean.log ===')
    experiments = parse_tune2_metrics('tune_stage2_clean.log')
    for name, maes, mses in experiments:
        print(f'  {name}: {len(maes)} samples, best MAE={min(maes):.3f}, best MSE={min(mses):.3f}')

    print('\n=== Generating plots ===')
    plot_stage1_loss(*vgg_loss, f'{OUTPUT_DIR}/fig_stage1_loss.png')
    plot_stage1_test(vgg_test, emac_test, f'{OUTPUT_DIR}/fig_stage1_test.png')
    plot_stage1_combined(vgg_test, emac_test, f'{OUTPUT_DIR}/fig_stage1_combined.png')
    plot_tune_comparison(experiments, f'{OUTPUT_DIR}/fig_tune2_comparison.png')
    plot_tune_series(experiments, f'{OUTPUT_DIR}/fig_tune2_series.png')
    print('\nAll figures generated.')


if __name__ == '__main__':
    main()
