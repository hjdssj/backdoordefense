"""
逻辑回归融合过程可视化

Fig1: 各指标权重（逻辑回归系数）
Fig2: 最重要的两个特征的 2D 散点图 + 决策边界
Fig3: 融合概率分布（后门 vs 干净）
Fig4: ROC 曲线对比（单指标 vs ML Fusion）
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))

import torch
import functools
original_torch_load = torch.load
@functools.wraps(original_torch_load)
def patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn import metrics


# ─────────────────────────────────────────────────────────────────────────────
# 加载数据
# ─────────────────────────────────────────────────────────────────────────────

ATTACK = 'badnet'
DATASET = 'cifar10'
PT_PATH = f'record/{ATTACK}_{DATASET}/saved/teco_enhanced_v2/detection_results.pt'

data = torch.load(PT_PATH)
bd  = data['bd_results']
cl  = data['clean_results']

FEATURE_NAMES = {
    'lp_flip_count':     'LP Flip Count',
    'lp_conf_drop':      'LP Conf Drop',
    'lp_kl_div':         'LP KL Div',
    'lp_entropy_change': 'LP Entropy',
    'hp_flip_count':     'HP Flip Count',
    'hp_conf_drop':      'HP Conf Drop',
    'hp_kl_div':         'HP KL Div',
    'hp_entropy_change': 'HP Entropy',
    'dct_flip_count':    'DCT Flip Count',
    'dct_conf_drop':     'DCT Conf Drop',
    'dct_kl_div':        'DCT KL Div',
    'dct_entropy_change':'DCT Entropy',
    's_flip_count':      'Spatial Flip',
    's_conf_drop':       'Spatial Conf',
    's_kl_div':          'Spatial KL',
    's_entropy_change':  'Spatial Entropy',
}
KEYS = list(FEATURE_NAMES.keys())
LABELS = list(FEATURE_NAMES.values())

X_bd    = np.column_stack([bd[k] for k in KEYS])
X_clean = np.column_stack([cl[k] for k in KEYS])
X = np.vstack([X_bd, X_clean])
y = np.array([1]*len(X_bd) + [0]*len(X_clean))

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

clf = LogisticRegression(max_iter=1000, C=1.0)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
proba = cross_val_predict(clf, X_norm, y, cv=skf, method='predict_proba')[:, 1]

# 用全量数据训练一次，拿权重用于可视化
clf.fit(X_norm, y)
weights = clf.coef_[0]

os.makedirs('figures', exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Fig1: 各指标权重
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))

colors = ['#d62728' if w > 0 else '#1f77b4' for w in weights]
bars = ax.barh(range(len(KEYS)), weights, color=colors, alpha=0.85, edgecolor='white')

ax.set_yticks(range(len(KEYS)))
ax.set_yticklabels(LABELS, fontsize=10)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Logistic Regression Coefficient', fontsize=11)
ax.set_title(f'Feature Weights in ML Fusion\n({ATTACK.upper()} / {DATASET.upper()})', fontsize=12)

# 颜色说明
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor='#d62728', label='Positive → higher = more likely backdoor'),
    Patch(facecolor='#1f77b4', label='Negative → higher = more likely clean'),
], fontsize=9, loc='lower right')

# 标注数值
for bar, w in zip(bars, weights):
    ax.text(w + (0.01 if w >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
            f'{w:.3f}', va='center', ha='left' if w >= 0 else 'right', fontsize=8)

plt.tight_layout()
plt.savefig('figures/fig_lr_weights.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: figures/fig_lr_weights.png')


# ─────────────────────────────────────────────────────────────────────────────
# Fig2: 最重要的两个特征的 2D 散点 + 决策边界
# ─────────────────────────────────────────────────────────────────────────────

# 取权重绝对值最大的两个特征
top2_idx = np.argsort(np.abs(weights))[-2:][::-1]
i1, i2 = top2_idx[0], top2_idx[1]

X2 = X_norm[:, [i1, i2]]
clf2 = LogisticRegression(max_iter=1000)
clf2.fit(X2, y)

fig, ax = plt.subplots(figsize=(8, 6))

# 子采样，避免点太多
np.random.seed(42)
bd_idx  = np.where(y == 1)[0]
cl_idx  = np.where(y == 0)[0]
sample_bd = np.random.choice(bd_idx,  min(500, len(bd_idx)),  replace=False)
sample_cl = np.random.choice(cl_idx,  min(500, len(cl_idx)),  replace=False)

ax.scatter(X2[sample_cl, 0], X2[sample_cl, 1],
           c='#1f77b4', alpha=0.4, s=15, label='Clean')
ax.scatter(X2[sample_bd, 0], X2[sample_bd, 1],
           c='#d62728', alpha=0.4, s=15, label='Backdoor')

# 决策边界
x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
Z = clf2.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

contour = ax.contourf(xx, yy, Z, levels=50, cmap='RdBu_r', alpha=0.3)
ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=1.5,
           linestyles='--')
plt.colorbar(contour, ax=ax, label='P(Backdoor)')

ax.set_xlabel(f'{LABELS[i1]} (normalized)', fontsize=11)
ax.set_ylabel(f'{LABELS[i2]} (normalized)', fontsize=11)
ax.set_title(f'Decision Boundary (Top-2 Features)\n{LABELS[i1]} vs {LABELS[i2]}', fontsize=12)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('figures/fig_lr_boundary.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: figures/fig_lr_boundary.png')


# ─────────────────────────────────────────────────────────────────────────────
# Fig3: 融合后的概率分布
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

proba_bd = proba[y == 1]
proba_cl = proba[y == 0]

bins = np.linspace(0, 1, 50)
ax.hist(proba_cl, bins=bins, alpha=0.6, color='#1f77b4', density=True, label='Clean')
ax.hist(proba_bd, bins=bins, alpha=0.6, color='#d62728', density=True, label='Backdoor')

# 最优阈值
fpr, tpr, thresholds = metrics.roc_curve(y, proba)
f1s = [metrics.f1_score(y, (proba > t).astype(int)) for t in thresholds]
best_thr = thresholds[np.argmax(f1s)]
ax.axvline(best_thr, color='black', linestyle='--', linewidth=1.5,
           label=f'Best threshold = {best_thr:.3f}')

auc_val = metrics.roc_auc_score(y, proba)
ax.set_xlabel('P(Backdoor)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(f'ML Fusion Output Distribution\nAUC = {auc_val:.4f}', fontsize=12)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('figures/fig_lr_proba.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: figures/fig_lr_proba.png')


# ─────────────────────────────────────────────────────────────────────────────
# Fig4: ROC 曲线对比（单指标 vs Fusion）
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 7))

ind_results = data['indicator_results']

# 挑 AUC 最高的 4 个单指标画出来
sorted_ind = sorted(ind_results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
colors_ind = ['#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']

for (key, m), c in zip(sorted_ind[:4], colors_ind):
    ax.plot(m['fpr'], m['tpr'], color=c, linewidth=1.2, alpha=0.8,
            label=f"{FEATURE_NAMES[key]} (AUC={m['roc_auc']:.3f})")

# ML Fusion ROC
fpr_ml, tpr_ml, _ = metrics.roc_curve(y, proba)
ax.plot(fpr_ml, tpr_ml, color='#d62728', linewidth=2.5,
        label=f'ML Fusion (AUC={auc_val:.4f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title(f'ROC Curve: Single Indicators vs ML Fusion\n({ATTACK.upper()} / {DATASET.upper()})', fontsize=12)
ax.legend(fontsize=9, loc='lower right')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])

plt.tight_layout()
plt.savefig('figures/fig_lr_roc.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: figures/fig_lr_roc.png')

print('\n全部保存到 figures/')
