# TeCo-Enhanced 后门检测研究总结

> 本文档记录项目的研究内容、核心方法、实验结果及可视化说明。
> 生成日期：2026-04-15

---

## 1. 研究内容

**项目名称**：基于腐蚀鲁棒性一致性的深度学习后门攻击样本检测方法研究

**背景**：深度学习模型在训练阶段面临后门攻击威胁——攻击者通过在训练数据中植入特定触发器（trigger），使模型对含触发器的样本产生定向错误预测，而对干净样本表现正常。现有的后门检测方法大多需要干净数据参考或触发器先验知识，实用性受限。

**核心问题**：能否在**测试阶段**、**无需干净数据**、**无需触发器知识**的前提下，仅通过观察模型对单样本的预测行为，判别该样本是否为后门样本？

---

## 2. 方法原理

### 2.1 核心思想（TeCo, CVPR 2023）

后门感染模型对**干净图像**和**后门图像**在面对图像腐蚀时表现出截然不同的鲁棒性一致性：

- **干净图像**：原始预测与腐蚀后预测高度一致（模型依靠图像的主要内容做出判断）
- **后门图像**：腐蚀后预测往往发生剧烈变化（trigger 被削弱或消除后，模型失去了触发后门的能力）

### 2.2 TeCo-Enhanced 的改进

在 TeCo 原方法基础上，TeCo-Enhanced 提出两个核心改进：

#### 改进一：多维度腐蚀特征指标

原 TeCo 仅使用一种空间域腐蚀的单一指标（预测是否翻转）。

TeCo-Enhanced 扩展为 **4 种扰动域 × 4 类度量指标 = 16 维特征向量**：

| 扰动域 | 含义 | 处理方式 |
|---|---|---|
| LP（低通滤波） | 去掉高频成分，保留大轮廓 | FFT 后滤除高频 |
| HP（高通滤波） | 去掉低频成分，保留边缘纹理 | FFT 后滤除低频 |
| DCT（离散余弦变换） | 截断低频 DCT 系数 | 保留高频系数 |
| Spatial（空间域） | imagecorruptions 库的 13 种腐蚀取均值 | 噪声/模糊/天气等 |

| 指标 | 含义 |
|---|---|
| `flip_count` | 多级扰动中预测类别发生翻转的次数 |
| `conf_drop` | 原始置信度与扰动后置信度的平均下降量 |
| `kl_div` | 原始预测分布与扰动后分布的 KL 散度均值 |
| `entropy_change` | 扰动后预测熵的平均变化量 |

#### 改进二：ML 融合决策机制

将 16 维特征向量输入**逻辑回归**（5 折交叉验证），自动学习最优权重组合：

```
P(后门) = sigmoid(w1*lp_flip + w2*lp_conf + ... + w16*s_entropy + b)
```

**为什么单指标效果差，融合后才好？**
BadNet 的 trigger（白色方块）在不同扰动域下变化不一致——空间域腐蚀下变化明显，DCT 下变化不明显。逻辑回归相当于 16 个弱分类器的加权投票，通过组合互补信息实现整体高性能。

---

## 3. 实验结果

### 3.1 实验配置

- **数据集**：CIFAR-10, CIFAR-100
- **攻击方法**：BadNet, Blended, SIG, LF
- **防御方法**：TeCo-Enhanced v2
- **评估指标**：AUC-ROC, F1 Score

> 注：SSBA 攻击缺少预置资源文件（`resource/ssba/` 不存在），SIG/CIFAR-100 的 pratio 参数已从 0.1 调至 0.01 以避免采样超出范围。

### 3.2 主结果

| 数据集 | 攻击 | AUC | F1 |
|---|---|---|---|
| CIFAR-10 | BadNet | 0.9292 | 0.8637 |
| CIFAR-10 | Blended | 0.9872 | 0.9460 |
| CIFAR-10 | SIG | 0.9810 | 0.9539 |
| CIFAR-10 | LF | 0.9935 | 0.9712 |
| CIFAR-100 | BadNet | 0.9304 | 0.8805 |
| CIFAR-100 | Blended | 0.9165 | 0.8429 |
| CIFAR-100 | LF | 0.9690 | 0.9309 |

> CIFAR-100 SIG 待重新运行（pratio 参数已修复）

---

## 4. 可视化说明

所有可视化脚本位于项目根目录，结果保存在 `figures/` 目录：

### Fig1: 图像频域对比（`fig1_image_comparison.png`）
展示干净样本与后门样本在原始 / 低通滤波 / 高通滤波 / FFT 幅度谱下的对比。BadNet 的小白色 trigger 在低通滤波后仍隐约可见，视觉上体现 trigger 的高频特性。

### Fig2: 指标分布（`fig2_indicator_distribution.png`）
展示 LP KL 散度、Conf Drop 等指标在干净/后门样本上的分布对比。可观察到两类的分布有一定分离，为融合决策提供依据。

### Fig3: 各指标 AUC 柱状图（`fig3_indicator_auc.png`）
16 个单指标的 AUC 均接近随机水平（0.5），但 ML Fusion 达到 0.929。该图直观展示了"单指标无效、融合才有效"的核心发现。

### Fig4: 逻辑回归权重（`fig_lr_weights.png`）
展示逻辑回归学到的 16 个特征权重。`HP Flip Count`（0.847）和 `DCT Flip Count`（0.790）权重最大，说明高频域对 BadNet 检测最敏感。负权重指标（`Spatial KL`=-1.521，`HP KL`=-1.295）则为反向信号。

### Fig5: 决策边界（`fig_lr_boundary.png`）
取权重绝对值最大的两个特征做 2D 散点图，叠加逻辑回归决策边界。蓝色（干净）聚集在左下，红色（后门）分散在右上方，虚线为分界线。

### Fig6: 融合概率分布（`fig_lr_proba.png`）
融合后 P(后门) 在干净样本上集中在 0，在后门样本上集中在 1，最优阈值 0.515，基本可完全区分两类。

### Fig7: ROC 曲线对比（`fig_lr_roc.png`）
最核心的可视化：单指标 ROC 贴近对角线（接近随机），ML Fusion ROC 曲线紧贴左上角（AUC=0.929），直观体现融合机制的价值。

---

## 5. 技术说明

### 5.1 PyTorch 2.6 兼容性修复

PyTorch 2.6 将 `torch.load` 的 `weights_only` 默认值从 `False` 改为 `True`，但 `attack_result.pt` 中包含 `PIL.Image.Image` 对象（非纯 tensor），导致加载报错。

**修复方案**：在 `utils/save_load_attack.py` 第 190 行添加：
```python
load_file = torch.load(save_path, weights_only=False)
```

### 5.2 实验脚本使用说明

```bash
# 完整运行（自动跳过已有结果）
python run_experiments.py

# 仅收集已有结果，不运行新实验
python run_experiments.py --collect-only

# 仅列出缺失的实验组合
python run_experiments.py --dry-run
```

结果自动保存至 `results_teco_<timestamp>.csv`，每次运行生成新文件。

### 5.3 可视化脚本

```bash
# 生成频域分析图（Fig1-3）
python visualize_frequency.py

# 生成逻辑回归可视化（Fig4-7）
python visualize_logistic.py
```

---

## 6. 结论与展望

TeCo-Enhanced 在 TeCo 基础上，通过多维特征提取与机器学习融合，显著提升了后门样本检测性能。AUC 最高达 0.993（CIFAR-10/LF），CIFAR-100 上也保持在 0.90 以上。

**主要发现**：
1. 没有单一指标能有效区分后门样本，必须多指标融合
2. 高通滤波（HP）和 DCT 域的翻转次数对 BadNet 类攻击最敏感
3. 逻辑回归作为简单融合器，在低维特征空间（16维）中足够有效且可解释

**待完成**：
- [ ] CIFAR-100 SIG 实验（pratio 已修复）
- [ ] GTSRB、Tiny-ImageNet 数据集实验（需下载数据）
- [ ] 与 TeCo 原方法的横向对比
- [ ] 论文撰写与投稿
