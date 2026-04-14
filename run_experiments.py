"""
TeCo-Enhanced 批量实验脚本
运行所有 数据集 × 攻击 组合，对比原始 TeCo 和 TeCo-v2
结果保存为 CSV，方便可视化

用法:
    python run_experiments.py                   # 全量运行
    python run_experiments.py --dry-run         # 只检查哪些组合缺失，不实际运行
    python run_experiments.py --collect-only    # 只收集已有结果到 CSV，不运行新实验
"""

import os
import sys
import argparse
import subprocess
import csv
from pathlib import Path
from datetime import datetime

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ─────────────────────────────────────────────────────────────────────────────
# 实验矩阵配置
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.absolute()

DATASETS = ['cifar10', 'cifar100']  # gtsrb/tiny 数据集未就绪，暂时跳过

ATTACKS = ['badnet', 'blended', 'sig', 'lf']  # ssba 缺少预置 resource 文件，暂时跳过

# 攻击脚本名称映射
ATTACK_SCRIPTS = {
    'badnet':  'attack/badnet_attack.py',
    'blended': 'attack/blended_attack.py',
    'sig':     'attack/sig_attack.py',
    'lf':      'attack/lf_attack.py',
    'ssba':    'attack/ssba_attack.py',
}

# 攻击 YAML 目录（先找 dataset.yaml，再找 default.yaml）
ATTACK_CONFIG_DIRS = {
    'badnet':  'config/attack/badnet',
    'blended': 'config/attack/blended',
    'sig':     'config/attack/sig',
    'lf':      'config/attack/lf',
    'ssba':    'config/attack/ssba',
}

DEFENSES = ['teco_enhanced_v2']

DEFENSE_SCRIPTS = {
    'teco':             'defense/teco/teco.py',
    'teco_enhanced_v2': 'defense/teco_enhanced/teco_enhanced_v2.py',
}

DEFENSE_CONFIG_DIRS = {
    'teco':             'config/defense/teco',
    'teco_enhanced_v2': 'config/defense/teco_enhanced',
}


# ─────────────────────────────────────────────────────────────────────────────
# 路径工具
# ─────────────────────────────────────────────────────────────────────────────

def find_yaml(config_dir: str, dataset: str) -> Path:
    """优先返回 <dataset>.yaml，退而求其次返回 default.yaml。"""
    d = ROOT / config_dir
    specific = d / f'{dataset}.yaml'
    if specific.exists():
        return specific
    default = d / 'default.yaml'
    if default.exists():
        return default
    raise FileNotFoundError(f"在 {d} 中找不到 {dataset}.yaml 或 default.yaml")


def attack_result_path(attack: str, dataset: str) -> Path:
    return ROOT / 'record' / f'{attack}_{dataset}' / 'attack_result.pt'


def teco_result_path(attack: str, dataset: str) -> Path:
    return ROOT / 'record' / f'{attack}_{dataset}' / 'saved' / 'teco' / 'defense_result_roc.pt'


def teco_v2_result_path(attack: str, dataset: str) -> Path:
    return ROOT / 'record' / f'{attack}_{dataset}' / 'saved' / 'teco_enhanced_v2' / 'detection_results.pt'


# ─────────────────────────────────────────────────────────────────────────────
# 运行子进程
# ─────────────────────────────────────────────────────────────────────────────

def run(cmd: list, label: str) -> bool:
    """运行命令，实时打印输出，返回是否成功。"""
    print(f"    $ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
    )
    for line in proc.stdout:
        print('    |', line, end='')
    proc.wait()
    if proc.returncode != 0:
        print(f"    [FAILED] {label} 返回码 {proc.returncode}")
        return False
    return True


def run_attack(attack: str, dataset: str) -> bool:
    save_folder = f'{attack}_{dataset}'
    record_dir = ROOT / 'record' / save_folder

    # 攻击脚本用 os.mkdir（无 exist_ok），目录存在会直接报错。
    # 如果目录存在但没有 attack_result.pt，说明是上次失败的残留，删掉重跑。
    if record_dir.exists() and not (record_dir / 'attack_result.pt').exists():
        import shutil
        print(f'    [CLEANUP] 删除不完整的目录: {record_dir}')
        shutil.rmtree(str(record_dir))

    yaml_path = find_yaml(ATTACK_CONFIG_DIRS[attack], dataset)
    cmd = [
        sys.executable, ATTACK_SCRIPTS[attack],
        '--yaml_path', str(yaml_path),
        '--dataset', dataset,
        '--dataset_path', '../data',
        '--save_folder_name', save_folder,
    ]
    return run(cmd, f'attack/{attack}/{dataset}')


def run_defense(defense: str, attack: str, dataset: str) -> bool:
    save_folder = f'{attack}_{dataset}'
    yaml_path = find_yaml(DEFENSE_CONFIG_DIRS[defense], dataset)
    cmd = [
        sys.executable, DEFENSE_SCRIPTS[defense],
        '--result_file', save_folder,
        '--yaml_path', str(yaml_path),
        '--dataset', dataset,
    ]
    return run(cmd, f'defense/{defense}/{attack}/{dataset}')


# ─────────────────────────────────────────────────────────────────────────────
# 指标提取
# ─────────────────────────────────────────────────────────────────────────────

def load_pt(path: Path) -> dict:
    if not HAS_TORCH:
        raise RuntimeError("需要 torch 才能读取 .pt 文件")
    return torch.load(str(path), weights_only=False, map_location='cpu')


def extract_teco_metrics(attack: str, dataset: str):
    """
    从 defense_result_roc.pt 提取:
      roc_auc  -> float
      f1_score -> float (数组中最大值)
    返回 (auc, f1) 或 (None, None)
    """
    p = teco_result_path(attack, dataset)
    if not p.exists():
        return None, None
    try:
        data = load_pt(p)
        auc = float(data['roc_auc'])
        f1_arr = data.get('f1_score', [0])
        import numpy as np
        f1 = float(np.max(f1_arr))
        return auc, f1
    except Exception as e:
        print(f"    [WARNING] 读取 {p} 失败: {e}")
        return None, None


def extract_teco_v2_metrics(attack: str, dataset: str):
    """
    从 detection_results.pt 提取:
      ml_metrics['roc_auc'] -> float
      ml_metrics['f1']      -> float
    返回 (auc, f1) 或 (None, None)
    """
    p = teco_v2_result_path(attack, dataset)
    if not p.exists():
        return None, None
    try:
        data = load_pt(p)
        ml = data.get('ml_metrics', {})
        auc = float(ml['roc_auc']) if 'roc_auc' in ml else None
        f1  = float(ml['best_f1'])  if 'best_f1' in ml else None
        return auc, f1
    except Exception as e:
        print(f"    [WARNING] 读取 {p} 失败: {e}")
        return None, None


EXTRACTORS = {
    'teco':             extract_teco_metrics,
    'teco_enhanced_v2': extract_teco_v2_metrics,
}

RESULT_PATHS = {
    'teco':             teco_result_path,
    'teco_enhanced_v2': teco_v2_result_path,
}


# ─────────────────────────────────────────────────────────────────────────────
# CSV 输出
# ─────────────────────────────────────────────────────────────────────────────

FIELDNAMES = ['dataset', 'attack', 'defense', 'auc', 'f1', 'timestamp']


def append_csv(csv_path: Path, row: dict):
    exists = csv_path.exists()
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_csv(csv_path: Path) -> list:
    if not csv_path.exists():
        return []
    with open(csv_path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def already_in_csv(rows: list, dataset: str, attack: str, defense: str) -> bool:
    for r in rows:
        if r['dataset'] == dataset and r['attack'] == attack and r['defense'] == defense:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='TeCo 批量实验')
    parser.add_argument('--dry-run', action='store_true',
                        help='只列出缺失的实验，不实际运行')
    parser.add_argument('--collect-only', action='store_true',
                        help='只收集已有 .pt 结果到 CSV，不运行新实验')
    parser.add_argument('--output', default='',
                        help='CSV 输出路径，默认 results_teco_<timestamp>.csv')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = Path(args.output) if args.output else ROOT / f'results_teco_{timestamp}.csv'

    print('=' * 80)
    print('TeCo-Enhanced 批量实验')
    print('=' * 80)
    print(f'数据集 : {DATASETS}')
    print(f'攻击   : {ATTACKS}')
    print(f'防御   : {DEFENSES}')
    print(f'CSV    : {csv_path}')
    print('=' * 80)

    existing = read_csv(csv_path)
    total = skipped = failed = 0

    for dataset in DATASETS:
        for attack in ATTACKS:
            # ── 1. 确保攻击结果存在 ────────────────────────────────────────
            ar = attack_result_path(attack, dataset)
            if not ar.exists():
                if args.dry_run or args.collect_only:
                    print(f'[MISSING] attack/{attack}/{dataset}')
                    continue
                print(f'\n>>> 运行攻击: {attack} / {dataset}')
                if not run_attack(attack, dataset):
                    failed += 1
                    continue

            # ── 2. 对每个防御 ──────────────────────────────────────────────
            for defense in DEFENSES:
                total += 1
                rp = RESULT_PATHS[defense](attack, dataset)

                # 已在 CSV 中则跳过
                if already_in_csv(existing, dataset, attack, defense):
                    print(f'[SKIP]    {dataset}/{attack}/{defense} (CSV 中已有)')
                    skipped += 1
                    continue

                # 结果文件不存在则运行防御
                if not rp.exists():
                    if args.dry_run or args.collect_only:
                        print(f'[MISSING] defense/{defense}/{attack}/{dataset}')
                        continue
                    print(f'\n>>> 运行防御: {defense} / {attack} / {dataset}')
                    if not run_defense(defense, attack, dataset):
                        failed += 1
                        continue

                # 提取指标
                auc, f1 = EXTRACTORS[defense](attack, dataset)
                if auc is None:
                    print(f'[NO METRIC] {dataset}/{attack}/{defense} — 无法读取指标')
                    failed += 1
                    continue

                row = {
                    'dataset':   dataset,
                    'attack':    attack,
                    'defense':   defense,
                    'auc':       f'{auc:.6f}',
                    'f1':        f'{f1:.6f}' if f1 is not None else '',
                    'timestamp': timestamp,
                }
                append_csv(csv_path, row)
                existing.append(row)
                f1_str = f'{f1:.4f}' if f1 is not None else 'N/A'
                print(f'[OK]      {dataset}/{attack}/{defense}  AUC={auc:.4f}  F1={f1_str}')

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    print('\n' + '=' * 80)
    print(f'完成: 总计={total}  跳过={skipped}  失败={failed}')
    print(f'CSV : {csv_path}')

    # 打印简单透视表
    all_rows = read_csv(csv_path)
    if all_rows:
        print('\n[AUC 汇总]')
        # 手动构建透视表（避免依赖 pandas）
        header_defenses = sorted({r['defense'] for r in all_rows})
        combo_map = {}
        for r in all_rows:
            key = (r['dataset'], r['attack'])
            combo_map.setdefault(key, {})[r['defense']] = r['auc']

        col_w = 20
        header = f"{'dataset+attack':<25}" + ''.join(f"{d:<{col_w}}" for d in header_defenses)
        print(header)
        print('-' * len(header))
        for (ds, atk), def_map in sorted(combo_map.items()):
            row_str = f"{ds+'/'+atk:<25}"
            for d in header_defenses:
                row_str += f"{def_map.get(d, 'N/A'):<{col_w}}"
            print(row_str)


if __name__ == '__main__':
    main()
