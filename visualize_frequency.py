"""
论文可视化脚本：展示低通/高通滤波对后门样本和干净样本的影响

生成以下图：
1. Fig1: 图像可视化 - 原始/低通/高通/频域幅度谱 对比（干净 vs 后门）
2. Fig2: 置信度曲线 - 随滤波强度变化（干净 vs 后门，均值±标准差）
3. Fig3: 指标分布 - KL散度/conf_drop 的分布对比（干净 vs 后门）
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

sys.path.append(os.path.dirname(__file__))

import torch
import functools
original_torch_load = torch.load
@functools.wraps(original_torch_load)
def patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

from scipy.fftpack import dct, idct
from utils.save_load_attack import load_attack_result


# ─────────────────────────────────────────────────────────────────────────────
# 滤波函数（复用 teco_enhanced_v2 的实现）
# ─────────────────────────────────────────────────────────────────────────────

def fft_filter(image_np, cutoff_ratio=0.1, filter_type='low'):
    """image_np: HxWxC uint8 -> uint8"""
    image = image_np.astype(np.float32)
    h, w, c = image.shape
    filtered = []
    for ch in range(c):
        f = np.fft.fft2(image[:, :, ch])
        fshift = np.fft.fftshift(f)
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((y - cy)**2 + (x - cx)**2)
        max_r = np.sqrt(cy**2 + cx**2)
        cutoff = cutoff_ratio * max_r
        if filter_type == 'low':
            mask = (radius > cutoff).astype(np.float32)
            mask[cy, cx] = 1.0
        else:  # high
            mask = (radius < cutoff).astype(np.float32)
        fshift_f = fshift * mask
        img_back = np.real(np.fft.ifft2(np.fft.ifftshift(fshift_f)))
        filtered.append(np.clip(img_back, 0, 255))
    return np.stack(filtered, axis=-1).astype(np.uint8)


def fft_magnitude(image_np):
    """返回 log 幅度谱（灰度，用于可视化）"""
    gray = image_np.mean(axis=2)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
    return magnitude


# ─────────────────────────────────────────────────────────────────────────────
# 加载数据
# ─────────────────────────────────────────────────────────────────────────────

def load_samples(attack, dataset, n_samples=5):
    """从 attack_result.pt 取前 n_samples 个后门样本和干净样本"""
    result = load_attack_result(f'record/{attack}_{dataset}/attack_result.pt')

    x_bd = result['bd_test']['x']
    x_clean = result['clean_test']['x']

    bd_imgs = []
    clean_imgs = []

    for i in range(n_samples):
        img = x_bd[i]
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        bd_imgs.append(img)

        img = x_clean[i]
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        clean_imgs.append(img)

    return bd_imgs, clean_imgs


# ─────────────────────────────────────────────────────────────────────────────
# Fig1: 图像对比（原始 / LP / HP / 频域幅度谱）
# ─────────────────────────────────────────────────────────────────────────────

def plot_image_comparison(bd_imgs, clean_imgs, save_path, cutoff=0.15):
    n = min(3, len(bd_imgs))  # 展示3个样本
    fig, axes = plt.subplots(4, n * 2, figsize=(4 * n, 8))
    fig.suptitle('Frequency Domain Analysis: Clean vs Backdoor Samples', fontsize=13, y=1.01)

    col_labels = []
    for i in range(n):
        col_labels += [f'Clean #{i+1}', f'Backdoor #{i+1}']

    row_labels = ['Original', f'Low-pass\n(cutoff={cutoff})', f'High-pass\n(cutoff={cutoff})', 'FFT Magnitude\n(log scale)']

    for col, (label, imgs) in enumerate(
        [(f'Clean #{i+1}', clean_imgs[i]) for i in range(n)] +
        [(f'Backdoor #{i+1}', bd_imgs[i]) for i in range(n)]
    ):
        # 按 clean0, bd0, clean1, bd1... 排列
        pass

    imgs_ordered = []
    for i in range(n):
        imgs_ordered.append(('Clean', clean_imgs[i]))
        imgs_ordered.append(('Backdoor', bd_imgs[i]))

    for col, (tag, img) in enumerate(imgs_ordered):
        lp = fft_filter(img, cutoff_ratio=cutoff, filter_type='low')
        hp = fft_filter(img, cutoff_ratio=cutoff, filter_type='high')
        mag = fft_magnitude(img)

        for row, (display_img, cmap) in enumerate([
            (img, None),
            (lp, None),
            (hp, None),
            (mag, 'viridis'),
        ]):
            ax = axes[row, col]
            if cmap:
                ax.imshow(display_img, cmap=cmap)
            else:
                ax.imshow(np.clip(display_img, 0, 255))
            ax.axis('off')
            if row == 0:
                color = '#d62728' if tag == 'Backdoor' else '#1f77b4'
                ax.set_title(f'{tag} #{col//2+1}', fontsize=9, color=color)

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9, rotation=0, labelpad=60, va='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig2: 置信度随滤波强度变化曲线
# ─────────────────────────────────────────────────────────────────────────────

def plot_confidence_curve(detection_pt_path, save_path):
    """用 lp_conf_drop 按 level 重建曲线（近似展示趋势）"""
    data = torch.load(detection_pt_path)
    bd = data['bd_results']
    cl = data['clean_results']

    metrics = {
        'LP KL Divergence':   ('lp_kl_div',  bd['lp_kl_div'],  cl['lp_kl_div']),
        'LP Conf Drop':       ('lp_conf_drop', bd['lp_conf_drop'], cl['lp_conf_drop']),
        'HP KL Divergence':   ('hp_kl_div',  bd['hp_kl_div'],  cl['hp_kl_div']),
        'Spatial KL Div':     ('s_kl_div',   bd['s_kl_div'],   cl['s_kl_div']),
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('Distribution of Detection Indicators: Clean vs Backdoor', fontsize=12)

    for ax, (title, (key, bd_vals, cl_vals)) in zip(axes, metrics.items()):
        # 截断极端值便于可视化
        p99 = np.percentile(np.concatenate([bd_vals, cl_vals]), 99)
        bd_clip = np.clip(bd_vals, 0, p99)
        cl_clip = np.clip(cl_vals, 0, p99)

        bins = np.linspace(0, p99, 50)
        ax.hist(cl_clip, bins=bins, alpha=0.6, label='Clean', color='#1f77b4', density=True)
        ax.hist(bd_clip, bins=bins, alpha=0.6, label='Backdoor', color='#d62728', density=True)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig3: 各指标 AUC 柱状图（消融）
# ─────────────────────────────────────────────────────────────────────────────

def plot_indicator_auc(detection_pt_path, save_path):
    data = torch.load(detection_pt_path)
    ind = data['indicator_results']

    names = {
        'lp_flip_count':  'LP Flip',
        'lp_conf_drop':   'LP Conf',
        'lp_kl_div':      'LP KL',
        'lp_entropy_change': 'LP Entropy',
        'hp_flip_count':  'HP Flip',
        'hp_conf_drop':   'HP Conf',
        'hp_kl_div':      'HP KL',
        'hp_entropy_change': 'HP Entropy',
        'dct_flip_count': 'DCT Flip',
        'dct_conf_drop':  'DCT Conf',
        'dct_kl_div':     'DCT KL',
        'dct_entropy_change': 'DCT Entropy',
        's_flip_count':   'Spatial Flip',
        's_conf_drop':    'Spatial Conf',
        's_kl_div':       'Spatial KL',
        's_entropy_change': 'Spatial Entropy',
    }

    keys = list(names.keys())
    aucs = [ind[k]['roc_auc'] for k in keys]
    labels = [names[k] for k in keys]

    colors = (['#1f77b4'] * 4 + ['#ff7f0e'] * 4 +
              ['#2ca02c'] * 4 + ['#d62728'] * 4)

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(keys)), aucs, color=colors, alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('AUC')
    ax.set_ylim(0.5, 1.0)
    ax.set_title('Per-Indicator Detection AUC (BadNet / CIFAR-10)', fontsize=12)
    ax.axhline(data['ml_metrics']['roc_auc'], color='black', linestyle='--',
               linewidth=1.5, label=f"ML Fusion: {data['ml_metrics']['roc_auc']:.4f}")
    ax.legend()

    # 标注每个柱的值
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=7)

    # 图例：颜色 = 域
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Low-pass FFT'),
        Patch(facecolor='#ff7f0e', label='High-pass FFT'),
        Patch(facecolor='#2ca02c', label='DCT'),
        Patch(facecolor='#d62728', label='Spatial'),
    ]
    ax.legend(handles=legend_elements + [
        plt.Line2D([0], [0], color='black', linestyle='--',
                   label=f"ML Fusion AUC={data['ml_metrics']['roc_auc']:.4f}")
    ], loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)

    ATTACK = 'badnet'
    DATASET = 'cifar10'
    DETECTION_PT = f'record/{ATTACK}_{DATASET}/saved/teco_enhanced_v2/detection_results.pt'

    print(f'加载样本: {ATTACK}/{DATASET}')
    bd_imgs, clean_imgs = load_samples(ATTACK, DATASET, n_samples=3)

    print('生成 Fig1: 图像频域对比...')
    plot_image_comparison(bd_imgs, clean_imgs,
                          save_path='figures/fig1_image_comparison.png')

    print('生成 Fig2: 指标分布...')
    plot_confidence_curve(DETECTION_PT,
                          save_path='figures/fig2_indicator_distribution.png')

    print('生成 Fig3: 各指标 AUC 柱状图...')
    plot_indicator_auc(DETECTION_PT,
                       save_path='figures/fig3_indicator_auc.png')

    print('\n所有图表已保存到 figures/ 目录')
