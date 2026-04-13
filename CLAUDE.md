# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BackdoorBench is a comprehensive benchmark of backdoor learning (NeurIPS 2022), providing implementations of 8 backdoor attack methods and 9 defense methods. It studies adversarial vulnerability of deep learning models in the training stage.

## Setup

```bash
sh ./sh/install.sh
mkdir -p record data data/cifar10 data/cifar100 data/gtsrb data/tiny
```

## Running Attacks

```bash
python ./attack/badnet_attack.py --yaml_path ../config/attack/badnet/cifar10.yaml --dataset cifar10 --dataset_path ../data --save_folder_name <folder_name>
```

Attack results are saved to `record/<folder_name>/attack_result.pt` - this file is required for subsequent defense runs.

## Running Defenses

Defenses require the attack result folder from a prior attack run:

```bash
python ./defense/ac/ac.py --result_file <attack_folder_name> --yaml_path ./config/defense/ac/cifar10.yaml --dataset cifar10
```

## Testing

```bash
# Test all attacks (1 epoch each)
sh ./sh/test_attack.sh

# Test all defenses
sh ./sh/test_defense.sh
```

## Code Architecture

### Attack Workflow (e.g., badnet_attack.py)
1. Load YAML config + CLI args, set random seed
2. Create clean train/test datasets via `dataset_and_transform_generate()`
3. Generate backdoor image/label transforms via `bd_attack_img_trans_generate()`, `bd_attack_label_trans_generate()`
4. Create poisoned datasets using `prepro_cls_DatasetBD` wrapper
5. Train model with `generate_cls_trainer()`
6. Save attack_result.pt via `save_attack_result()`

### Defense Workflow (e.g., ac.py)
1. Load attack result via `load_attack_result()` to get poisoned model + data
2. Apply defense-specific detection/mitigation
3. Evaluate and save defense_result.pt with metrics (ASR, ACC, RA)

### Key Shared Utilities (utils/aggregate_block/)
- `model_trainer_generate.py` - `generate_cls_model()` for model creation, `generate_cls_trainer()` for training loop
- `dataset_and_transform_generate.py` - `dataset_and_transform_generate()` for data loading, `get_num_classes()`, `get_input_shape()`
- `bd_attack_generate.py` - backdoor image/label transform generation
- `train_settings_generate.py` - optimizer and scheduler creation

### Backdoor Dataset Wrapper (utils/bd_dataset.py)
`prepro_cls_DatasetBD` wraps clean datasets with poison indicators, applying backdoor transforms only to samples where `poison_idx == 1`.

### Config System
YAML files in `config/attack/<method>/` and `config/defense/<method>/` store default hyperparameters. CLI args override YAML values. Each dataset (cifar10, cifar100, gtsrb, tiny) has its own YAML.

## Supported Methods

**Attacks**: badnet, blended, lc (label consistent), sig, lf (low frequency), ssba, inputaware, wanet
**Defenses**: ft, fp, ac, spectral, abl, nad, nc, anp, dbd, teco
**Models**: preactresnet18, vgg19, densenet161, mobilenet_v3_large, efficientnet_b3
**Datasets**: cifar10, cifar100, gtsrb, tiny (ImageNet subset)

## TeCo (CVPR 2023)

TeCo is a **test-time trigger sample detection** method based on Corruption Robustness Consistency. It detects backdoor samples without needing clean data or trigger knowledge.

**Core Idea**: Backdoor-infected models show consistent predictions across image corruptions for clean images, but inconsistent (discrepant) predictions for trigger samples.

**Install additional dependency**:
```bash
pip install imagecorruptions
```

**Usage**:
```bash
# 1. Train backdoor model
python ./attack/badnet_attack.py --yaml_path ../config/attack/badnet/cifar10.yaml --dataset cifar10 --dataset_path ../data --save_folder_name badnet_0_1

# 2. TeCo detection
python ./defense/teco/teco.py --result_file badnet_0_1 --yaml_path ./config/defense/teco/cifar10.yaml --dataset cifar10
```

**Output**: `record/<result_file>/saved/teco/defense_result_roc.pt` contains:
- `roc_auc`: AUROC score for detecting trigger samples
- `fpr`, `tpr`: ROC curve data
- `f1_score`: F1 scores at different thresholds
