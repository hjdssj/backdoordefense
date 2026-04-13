# TeCo-Enhanced: Cross-Domain Backdoor Detection

## 一、研究背景

### 1.1 TeCo 原始方法 (CVPR 2023)

TeCo (Test-time Corruption robustness Consistency) 是一种无需训练、无需干净数据的 backdoor 检测方法。

**核心发现**：backdoor 感染模型对干净样本和 trigger 样本在不同图像腐蚀下表现出不同的鲁棒性一致性：
- **干净样本**：各种腐蚀下预测稳定一致
- **trigger 样本**：腐蚀干扰 trigger 激活，预测发生跳转

**方法**：对图像施加 15 种图像腐蚀 × 5 级严重度，通过计算"预测首次变化的严重程度"的标准差来检测 trigger 样本。

### 1.2 改进动机

原始 TeCo 只在空间域进行图像腐蚀。本文提出扩展到**频域分析**，同时考虑空间域和频率域的扰动。

---

## 二、方法设计

### 2.1 整体框架

```
输入图像
    │
    ├─→ 空间域扰动序列 ─→ P_s (15种腐蚀 × 5级)
    │                              ↓
    │                    计算: flip, kl_div, entropy_change
    │
    ├─→ 低通频域扰动 ─→ P_lp (FFT低通 × 5级cutoff)
    │                              ↓
    │                    计算: flip, kl_div, entropy_change
    │
    ├─→ 高通频域扰动 ─→ P_hp (FFT高通 × 5级cutoff)
    │                              ↓
    │                    计算: flip, kl_div, entropy_change
    │
    └─→ DCT频域扰动 ─→ P_dct (DCT × 5级cutoff)
                               ↓
                         计算: flip, kl_div, entropy_change
```

### 2.2 频域变换

#### FFT 低通滤波 (Low-pass)
- 去除低频分量，保留高频（边缘、纹理）
- 原理：trigger（如 SIG 的高频正弦波）被抑制

#### FFT 高通滤波 (High-pass)
- 去除高频分量，保留低频（整体亮度、颜色）
- 原理：trigger（如 LF 的低频信号）被抑制

#### DCT 滤波
- 离散余弦变换，对局部模式更敏感
- 原理：适合分析 patch 类 trigger 的局部特征

### 2.3 检测指标

| 指标 | 说明 | 公式 |
|-----|------|-----|
| flip_count | 翻转次数 | 预测类别发生变化的扰动级别数 |
| kl_div | KL散度 | D(P_orig \|\| P_perturbed) |
| entropy_change | 熵变化 | H(P_perturbed) - H(P_orig) |
| conf_drop | 置信度下降 | max(P_orig) - max(P_perturbed) |

### 2.4 多指标融合

使用逻辑回归在验证集上进行 5 折交叉验证，学习最优指标组合权重。

---

## 三、实验结果

### 3.1 数据集和攻击

- **数据集**：CIFAR-10
- **模型**：PreActResNet18
- **测试攻击**：
  - **SIG** (Signal Injection Attack)：高频正弦波 trigger
  - **LF** (Low Frequency Attack)：低频信号 trigger
  - **BadNet**：3×3 白色 patch trigger

### 3.2 v1 结果（基础频域指标）

| 攻击 | 最佳指标 | AUC |
|-----|---------|-----|
| SIG | conf_drop_f (低通置信度下降) | 0.775 |
| LF | conf_drop_f_hp (高通置信度下降) | 0.727 |
| BadNet | flip_count_f (低通翻转次数) | 0.577 |

**发现**：不同频率特性的攻击需要不同的滤波策略

### 3.3 v2 结果（增强指标 + ML融合）

#### SIG 攻击

| 方法 | AUC | F1 |
|-----|-----|-----|
| LP KL Div (最佳单一) | 0.777 | 0.799 |
| ML Fusion (All, 5-CV) | **0.981** | **0.954** |

#### BadNet 攻击

| 方法 | AUC | F1 |
|-----|-----|-----|
| DCT Flip Count (最佳单一) | 0.590 | 0.666 |
| ML Fusion (All, 5-CV) | **0.912** | **0.845** |

**注意**：ML Fusion 使用 5 折交叉验证评估，结果为真实泛化性能。

---

## 四、关键发现

### 4.1 频率域分析的价值

| 攻击类型 | Trigger 频率特性 | 最有效滤波 |
|---------|-----------------|-----------|
| SIG | 高频正弦波 | 低通滤波 |
| LF | 低频信号 | 高通滤波 |
| BadNet | 高频 patch | DCT/低通 |

**结论**：频域分析能有效区分不同频率特性的 trigger，针对性选择滤波策略可提升检测效果。

### 4.2 KL 散度的优势

KL 散度比置信度下降更好地捕捉预测分布变化：
- KL 考虑了完整概率分布
- 对弱 trigger 更敏感

### 4.3 ML 融合的效果

多指标融合显著提升检测性能：
- SIG: 0.777 → 0.981 (+26%)
- BadNet: 0.590 → 0.912 (+55%)

---

## 五、代码结构

```
defense/teco_enhanced/
├── __init__.py
├── teco_enhanced.py      # v1: 基础 LP/HP + flip 指标
├── teco_enhanced_v2.py   # v2: KL散度 + DCT + ML融合
└── ...

config/defense/teco_enhanced/
├── default.yaml
├── cifar10.yaml
├── cifar100.yaml
├── gtsrb.yaml
└── tiny.yaml
```

---

## 六、使用方法

### 6.1 运行 v1 检测

```bash
python ./defense/teco_enhanced/teco_enhanced.py \
  --result_file <attack_folder> \
  --yaml_path ./config/defense/teco_enhanced/cifar10.yaml \
  --dataset cifar10
```

### 6.2 运行 v2 检测

```bash
python ./defense/teco_enhanced/teco_enhanced_v2.py \
  --result_file <attack_folder> \
  --yaml_path ./config/defense/teco_enhanced/cifar10.yaml \
  --dataset cifar10
```

### 6.3 先运行攻击生成模型

```bash
# SIG 攻击
python ./attack/sig_attack.py \
  --yaml_path ../config/attack/sig/cifar10.yaml \
  --dataset cifar10 --dataset_path ../data \
  --save_folder_name sig_test

# LF 攻击
python ./attack/lf_attack.py \
  --yaml_path ../config/attack/lf/cifar10.yaml \
  --dataset cifar10 --dataset_path ../data \
  --save_folder_name lf_test

# BadNet 攻击
python ./attack/badnet_attack.py \
  --yaml_path ../config/attack/badnet/cifar10.yaml \
  --dataset cifar10 --dataset_path ../data \
  --save_folder_name badnet_test
```

---

## 七、未来改进方向

1. **更多频段分析**：将频谱分成更细的 5-10 个频段分别分析
2. **幅度/相位分离扰动**：FFT 后分别扰动幅度和相位
3. **自适应滤波选择**：根据检测结果自动选择最有效的滤波策略
4. **深度特征分析**：分析模型中间层激活的频率特性

---

## 八、参考文献

1. Liu, X. et al. (2023). TeCo: Detecting Backdoors During the Inference Stage Based on Corruption Robustness Consistency. CVPR.
2. Wu, B. et al. (2022). BackdoorBench: A Comprehensive Benchmark of Backdoor Learning. NeurIPS.
3. Zeng, Y. et al. (2021). Rethinking the Backdoor Attacks' Triggers: A Frequency Perspective. ICCV.
