"""
TeCo-Enhanced v2: Advanced Cross-Domain Robustness Consistency Detection

Improvements over v1:
1. Multi-band frequency analysis (5 frequency bands)
2. Better metrics: KL divergence, entropy change, top-K probability change
3. Separate amplitude/phase perturbation
4. ML-based indicator fusion
"""

import argparse
import os
import sys
import functools
import inspect

sys.path.append('../')
sys.path.append(os.getcwd())

import numpy as np
import torch
import yaml
from PIL import Image
from scipy.fftpack import dct, idct
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import auc
from tqdm import tqdm

# PyTorch 2.11+ torch.load compatibility patch
original_torch_load = torch.load
@functools.wraps(original_torch_load)
def patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# scipy >= 1.14 gaussian_filter multichannel patch
import scipy.ndimage as ndi
_gauss_sig = inspect.signature(ndi.gaussian_filter)
_gauss_accepts_multichannel = 'multichannel' in _gauss_sig.parameters
_original_gauss = ndi.gaussian_filter
def patched_gauss(input, sigma, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0, multichannel=None):
    kwargs = {'input': input, 'sigma': sigma, 'order': order, 'output': output,
              'mode': mode, 'cval': cval, 'truncate': truncate}
    if _gauss_accepts_multichannel and multichannel is not None:
        kwargs['multichannel'] = multichannel
    return _original_gauss(**kwargs)
ndi.gaussian_filter = patched_gauss

from imagecorruptions import corrupt
from utils.aggregate_block.dataset_and_transform_generate import get_transform
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.save_load_attack import load_attack_result


# ============================================================
# Spatial Domain Transforms
# ============================================================

SPATIAL_CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'motion_blur', 'zoom_blur',
    'snow', 'fog', 'brightness', 'contrast',
    'elastic_transform', 'pixelate', 'jpeg_compression'
]


def spatial_transform(image, corruption_name, severity):
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = corrupt(image, corruption_name=corruption_name, severity=severity)
    return Image.fromarray(image)


# ============================================================
# Frequency Domain Transforms
# ============================================================

def fft_filter(image, cutoff_ratio=0.1, filter_type='low', preserve_dc=True):
    """
    Apply FFT-based frequency filtering.

    Args:
        image: PIL Image or numpy array
        cutoff_ratio: cutoff radius ratio (0-1)
        filter_type: 'low', 'high', or 'band' (band = remove middle frequencies)
        preserve_dc: keep DC component
    """
    if isinstance(image, Image.Image):
        image = np.array(image).astype(np.float32)
    else:
        image = image.astype(np.float32)

    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)

    h, w, c = image.shape
    filtered_channels = []

    for ch in range(c):
        f = np.fft.fft2(image[:, :, ch])
        fshift = np.fft.fftshift(f)

        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        max_radius = np.sqrt(center_y**2 + center_x**2)
        cutoff_radius = cutoff_ratio * max_radius

        if filter_type == 'low':
            mask = (radius > cutoff_radius).astype(np.float32)
            if preserve_dc:
                mask[center_y, center_x] = 1.0
        elif filter_type == 'high':
            mask = (radius < cutoff_radius).astype(np.float32)
            if preserve_dc:
                mask[center_y, center_x] = 0.0
        else:  # band: remove middle frequencies
            inner_radius = cutoff_ratio * 0.5 * max_radius
            mask = np.ones_like(radius, dtype=np.float32)
            mask[(radius > inner_radius) & (radius < cutoff_radius)] = 0.0

        fshift_filtered = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.real(np.fft.ifft2(f_ishift))
        img_back = np.clip(img_back, 0, 255)
        filtered_channels.append(img_back)

    result = np.stack(filtered_channels, axis=-1).astype(np.uint8)
    return Image.fromarray(result)


def dct_filter(image, cutoff_ratio=0.3):
    """
    Apply DCT-based filtering. DCT is better for local patterns.
    """
    if isinstance(image, Image.Image):
        image = np.array(image).astype(np.float32)
    else:
        image = image.astype(np.float32)

    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)

    h, w, c = image.shape
    filtered_channels = []

    # DCT cutoff: keep top 'cutoff_ratio' of coefficients
    cutoff_h = int(h * cutoff_ratio)
    cutoff_w = int(w * cutoff_ratio)

    for ch in range(c):
        # DCT
        dct_coeffs = dct(dct(image[:, :, ch].T, type=2, norm='ortho').T, type=2, norm='ortho')

        # Zero out low-frequency coefficients
        dct_filtered = np.zeros_like(dct_coeffs)
        dct_filtered[cutoff_h:, :] = dct_coeffs[cutoff_h:, :]
        dct_filtered[:, cutoff_w:] = dct_coeffs[:, cutoff_w:]

        # Inverse DCT
        img_back = idct(idct(dct_filtered.T, type=2, norm='ortho').T, type=2, norm='ortho')
        img_back = np.clip(img_back, 0, 255)
        filtered_channels.append(img_back)

    result = np.stack(filtered_channels, axis=-1).astype(np.uint8)
    return Image.fromarray(result)


def fft_amplitude_perturb(image, drop_ratio=0.3):
    """
    Only perturb amplitude, keep phase.
    """
    if isinstance(image, Image.Image):
        image = np.array(image).astype(np.float32)
    else:
        image = image.astype(np.float32)

    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)

    h, w, c = image.shape
    filtered_channels = []

    for ch in range(c):
        f = np.fft.fft2(image[:, :, ch])
        amplitude = np.abs(f)
        phase = np.angle(f)

        # Perturb amplitude: reduce by drop_ratio
        amplitude_perturbed = amplitude * (1 - drop_ratio)

        # Reconstruct
        f_perturbed = amplitude_perturbed * np.exp(1j * phase)
        f_ishift = np.fft.ifftshift(f_perturbed)
        img_back = np.real(np.fft.ifft2(f_ishift))
        img_back = np.clip(img_back, 0, 255)
        filtered_channels.append(img_back)

    result = np.stack(filtered_channels, axis=-1).astype(np.uint8)
    return Image.fromarray(result)


def fft_phase_perturb(image, shift_ratio=0.5):
    """
    Only perturb phase, keep amplitude.
    """
    if isinstance(image, Image.Image):
        image = np.array(image).astype(np.float32)
    else:
        image = image.astype(np.float32)

    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)

    h, w, c = image.shape
    filtered_channels = []

    for ch in range(c):
        f = np.fft.fft2(image[:, :, ch])
        amplitude = np.abs(f)
        phase = np.angle(f)

        # Perturb phase: add random shift
        phase_shift = np.random.randn(h, w) * shift_ratio * np.pi
        phase_perturbed = phase + phase_shift

        # Reconstruct
        f_perturbed = amplitude * np.exp(1j * phase_perturbed)
        f_ishift = np.fft.ifftshift(f_perturbed)
        img_back = np.real(np.fft.ifft2(f_ishift))
        img_back = np.clip(img_back, 0, 255)
        filtered_channels.append(img_back)

    result = np.stack(filtered_channels, axis=-1).astype(np.uint8)
    return Image.fromarray(result)


# ============================================================
# Detection Metrics
# ============================================================

def kl_divergence(p, q, eps=1e-10):
    """KL divergence: D(p||q)"""
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * np.log(p / q))


def entropy(p, eps=1e-10):
    """Shannon entropy"""
    p = np.clip(p, eps, 1)
    return -np.sum(p * np.log(p))


def compute_detection_metrics(P_perturbed, P_orig, num_levels):
    """
    Compute comprehensive detection metrics.

    Returns:
        flip: class flip indicator
        flip_count: number of levels with flip
        first_flip_level: first level where flip occurred
        conf_drop_first: confidence drop at first level
        conf_drop_avg: average confidence drop
        kl_div_avg: average KL divergence
        entropy_change_avg: average entropy change
        top2_change: change in top-2 prediction difference
    """
    P_orig = np.array(P_orig)
    P_perturbed = np.array(P_perturbed)

    orig_class = np.argmax(P_orig)
    orig_conf = P_orig[orig_class]

    perturbed_classes = np.argmax(P_perturbed, axis=1)
    flips = perturbed_classes != orig_class

    flip = int(any(flips))
    flip_count = int(np.sum(flips))
    first_flip_level = num_levels + 1 if not any(flips) else int(np.argmax(flips) + 1)

    # Confidence drop
    perturbed_first = P_perturbed[0]
    conf_drop_first = float(P_orig[orig_class] - perturbed_first[orig_class]) if orig_class < len(perturbed_first) else 0.0

    perturbed_max = np.max(P_perturbed, axis=1)
    conf_drop_avg = float(np.mean(orig_conf - perturbed_max))

    # KL divergence (avg across levels)
    kl_divs = [kl_divergence(P_orig, P_perturbed[l]) for l in range(num_levels)]
    kl_div_avg = float(np.mean(kl_divs))

    # Entropy change
    entropy_orig = entropy(P_orig)
    entropies_perturbed = [entropy(P_perturbed[l]) for l in range(num_levels)]
    entropy_change_avg = float(np.mean([entropies_perturbed[l] - entropy_orig for l in range(num_levels)]))

    # Top-2 change
    top2_orig = np.sort(P_orig)[-2]
    top2_first = np.sort(P_perturbed[0])[-2]
    top2_change = float(top2_first - top2_orig)

    return {
        'flip': flip,
        'flip_count': flip_count,
        'first_flip_level': first_flip_level,
        'conf_drop_first': conf_drop_first,
        'conf_drop_avg': conf_drop_avg,
        'kl_div_avg': kl_div_avg,
        'entropy_change_avg': entropy_change_avg,
        'top2_change': top2_change
    }


def get_frequency_trajectory_v2(image, model, device, num_levels=5, filter_type='low'):
    """
    Generate prediction trajectory with frequency perturbations.

    Args:
        image: PIL Image
        model: classifier
        device: cuda/cpu
        num_levels: number of perturbation levels
        filter_type: 'low', 'high', 'band'

    Returns:
        predictions: list of probability arrays
    """
    predictions = []

    if filter_type == 'low':
        cutoff_range = (0.05, 0.5)
        min_cutoff, max_cutoff = cutoff_range
        for level in range(1, num_levels + 1):
            cutoff = max_cutoff - (max_cutoff - min_cutoff) * (level - 1) / (num_levels - 1)
            perturbed = fft_filter(image, cutoff_ratio=cutoff, filter_type='low')
            img_t = get_transform('cifar10', 32, 32)(perturbed).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(img_t)
                prob = torch.softmax(out, dim=1)[0].cpu().numpy()
            predictions.append(prob)

    elif filter_type == 'high':
        cutoff_range = (0.05, 0.5)
        min_cutoff, max_cutoff = cutoff_range
        for level in range(1, num_levels + 1):
            cutoff = max_cutoff - (max_cutoff - min_cutoff) * (level - 1) / (num_levels - 1)
            perturbed = fft_filter(image, cutoff_ratio=cutoff, filter_type='high')
            img_t = get_transform('cifar10', 32, 32)(perturbed).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(img_t)
                prob = torch.softmax(out, dim=1)[0].cpu().numpy()
            predictions.append(prob)

    elif filter_type == 'dct':
        for level in range(1, num_levels + 1):
            cutoff = 0.5 - (0.4) * (level - 1) / (num_levels - 1)
            perturbed = dct_filter(image, cutoff_ratio=cutoff)
            img_t = get_transform('cifar10', 32, 32)(perturbed).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(img_t)
                prob = torch.softmax(out, dim=1)[0].cpu().numpy()
            predictions.append(prob)

    return predictions


def get_spatial_trajectory(image, model, device, num_levels=5, num_corruptions=5):
    """Get spatial domain prediction trajectory."""
    predictions = []

    for level in range(1, num_levels + 1):
        level_preds = []
        for corruption in SPATIAL_CORRUPTIONS[:num_corruptions]:
            perturbed = spatial_transform(image, corruption, level)
            img_t = get_transform('cifar10', 32, 32)(perturbed).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(img_t)
                prob = torch.softmax(out, dim=1)[0].cpu().numpy()
            level_preds.append(prob)
        predictions.append(np.mean(level_preds, axis=0))

    return predictions


# ============================================================
# Main Detection Function
# ============================================================

def detect_backdoor_samples_v2(model, data_loader, device, num_levels=5):
    """
    Detect backdoor samples using advanced cross-domain metrics.
    """
    results = {
        'indices': [],
        # Low-pass FFT metrics
        'lp_flip': [], 'lp_flip_count': [], 'lp_first_flip': [],
        'lp_conf_drop': [], 'lp_kl_div': [], 'lp_entropy_change': [],
        # High-pass FFT metrics
        'hp_flip': [], 'hp_flip_count': [], 'hp_first_flip': [],
        'hp_conf_drop': [], 'hp_kl_div': [], 'hp_entropy_change': [],
        # DCT metrics
        'dct_flip': [], 'dct_flip_count': [], 'dct_first_flip': [],
        'dct_conf_drop': [], 'dct_kl_div': [], 'dct_entropy_change': [],
        # Spatial metrics
        's_flip': [], 's_flip_count': [], 's_first_flip': [],
        's_conf_drop': [], 's_kl_div': [], 's_entropy_change': [],
        'predictions': []
    }

    model.eval()

    for idx, (inputs, labels) in enumerate(tqdm(data_loader, desc="Detecting")):
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            original_preds = probs.cpu().numpy()

        for i in range(inputs.size(0)):
            sample_idx = idx * data_loader.batch_size + i
            img = inputs[i].cpu()

            # Convert to PIL
            img_unnorm = img * 0.5 + 0.5
            img_unnorm = torch.clamp(img_unnorm, 0, 1)
            img_pil = Image.fromarray((img_unnorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

            P_orig = original_preds[i]

            # Get trajectories for each domain
            P_lp = get_frequency_trajectory_v2(img_pil, model, device, num_levels, 'low')
            P_hp = get_frequency_trajectory_v2(img_pil, model, device, num_levels, 'high')
            P_dct = get_frequency_trajectory_v2(img_pil, model, device, num_levels, 'dct')
            P_s = get_spatial_trajectory(img_pil, model, device, num_levels)

            # Compute metrics
            lp_m = compute_detection_metrics(P_lp, P_orig, num_levels)
            hp_m = compute_detection_metrics(P_hp, P_orig, num_levels)
            dct_m = compute_detection_metrics(P_dct, P_orig, num_levels)
            s_m = compute_detection_metrics(P_s, P_orig, num_levels)

            results['indices'].append(sample_idx)

            # LP
            results['lp_flip'].append(lp_m['flip'])
            results['lp_flip_count'].append(lp_m['flip_count'])
            results['lp_first_flip'].append(lp_m['first_flip_level'])
            results['lp_conf_drop'].append(lp_m['conf_drop_avg'])
            results['lp_kl_div'].append(lp_m['kl_div_avg'])
            results['lp_entropy_change'].append(lp_m['entropy_change_avg'])

            # HP
            results['hp_flip'].append(hp_m['flip'])
            results['hp_flip_count'].append(hp_m['flip_count'])
            results['hp_first_flip'].append(hp_m['first_flip_level'])
            results['hp_conf_drop'].append(hp_m['conf_drop_avg'])
            results['hp_kl_div'].append(hp_m['kl_div_avg'])
            results['hp_entropy_change'].append(hp_m['entropy_change_avg'])

            # DCT
            results['dct_flip'].append(dct_m['flip'])
            results['dct_flip_count'].append(dct_m['flip_count'])
            results['dct_first_flip'].append(dct_m['first_flip_level'])
            results['dct_conf_drop'].append(dct_m['conf_drop_avg'])
            results['dct_kl_div'].append(dct_m['kl_div_avg'])
            results['dct_entropy_change'].append(dct_m['entropy_change_avg'])

            # Spatial
            results['s_flip'].append(s_m['flip'])
            results['s_flip_count'].append(s_m['flip_count'])
            results['s_first_flip'].append(s_m['first_flip_level'])
            results['s_conf_drop'].append(s_m['conf_drop_avg'])
            results['s_kl_div'].append(s_m['kl_div_avg'])
            results['s_entropy_change'].append(s_m['entropy_change_avg'])

            results['predictions'].append(P_orig)

    # Convert to arrays
    for key in results:
        if key != 'indices':
            results[key] = np.array(results[key])
    results['predictions'] = np.array(results['predictions'])

    return results


def compute_roc_metrics(scores, labels):
    """Compute ROC-AUC and F1 metrics."""
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    f1_scores = []
    for th in thresholds:
        pred = (scores > th).astype(int)
        f1 = metrics.f1_score(labels, pred)
        f1_scores.append(f1)

    best_idx = np.argmax(f1_scores)
    return {
        'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds,
        'roc_auc': roc_auc,
        'best_f1': f1_scores[best_idx],
        'best_threshold': thresholds[best_idx]
    }


def train_ml_fusion(bd_results, clean_results, feature_keys, method='cv'):
    """
    Train logistic regression on multiple indicators with proper evaluation.

    Args:
        bd_results: backdoor sample results dict
        clean_results: clean sample results dict
        feature_keys: list of indicator keys to use
        method: 'cv' for cross-validation, 'holdout' for train/test split

    Returns:
        proba: predicted probabilities (cross-validated predictions)
        clf: trained classifier (for reference, not used for final eval)
        metrics: evaluation metrics using cross-validated predictions
    """
    X_bd = np.column_stack([bd_results[k] for k in feature_keys])
    X_clean = np.column_stack([clean_results[k] for k in feature_keys])
    X = np.vstack([X_bd, X_clean])

    y = np.concatenate([np.ones(len(X_bd)), np.zeros(len(X_clean))])

    # Normalize features
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8
    X_norm = (X - mean) / std

    clf = LogisticRegression(max_iter=1000, C=1.0)

    if method == 'cv':
        # 5-fold cross-validation: train on 4/5, predict on 1/5
        # This gives unbiased estimates of performance
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        proba = cross_val_predict(clf, X_norm, y, cv=skf, method='predict_proba')[:, 1]

        # Compute metrics using cross-validated predictions
        metrics = compute_roc_metrics(proba, y)
        return proba, clf, metrics

    elif method == 'holdout':
        # 80/20 train/test split
        n_train = int(0.8 * len(X))
        indices = np.random.RandomState(42).permutation(len(X))
        train_idx, test_idx = indices[:n_train], indices[n_train:]

        clf.fit(X_norm[train_idx], y[train_idx])
        proba_test = clf.predict_proba(X_norm[test_idx])[:, 1]

        metrics = compute_roc_metrics(proba_test, y[test_idx])
        return proba_test, clf, metrics


# ============================================================
# CLI Interface
# ============================================================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--result_file', type=str, required=True)
    parser.add_argument('--yaml_path', type=str, default='./config/defense/teco_enhanced/cifar10.yaml')
    parser.add_argument('--num_levels', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    with open(args.yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    config.update({k: v for k, v in args.__dict__.items() if v is not None})
    args.__dict__ = config

    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.input_height, args.input_width, args.input_channel = 32, 32, 3
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.input_height, args.input_width, args.input_channel = 32, 32, 3
    elif args.dataset == 'gtsrb':
        args.num_classes = 43
        args.input_height, args.input_width, args.input_channel = 32, 32, 3
    elif args.dataset == 'tiny':
        args.num_classes = 200
        args.input_height, args.input_width, args.input_channel = 64, 64, 3

    save_path = 'record/' + args.result_file

    print(f"[1] Loading attack result from {save_path}/attack_result.pt")
    result = load_attack_result(save_path + '/attack_result.pt')

    print("[2] Loading model")
    model = generate_cls_model(args.model or 'preactresnet18', args.num_classes)
    model.load_state_dict(result['model'])
    model.to(args.device)
    model.eval()

    print("[3] Preparing data")
    tran = get_transform(args.dataset, args.input_height, args.input_width, train=False)

    x_bd = result['bd_test']['x']
    y_bd = result['bd_test']['y']
    bd_dataset = prepro_cls_DatasetBD(
        full_dataset_without_transform=list(zip(x_bd, y_bd)),
        poison_idx=np.zeros(len(x_bd)),
        bd_image_pre_transform=None, bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran, ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    bd_loader = torch.utils.data.DataLoader(bd_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)

    x_clean = result['clean_test']['x']
    y_clean = result['clean_test']['y']
    clean_dataset = prepro_cls_DatasetBD(
        full_dataset_without_transform=list(zip(x_clean, y_clean)),
        poison_idx=np.zeros(len(x_clean)),
        bd_image_pre_transform=None, bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran, ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    clean_loader = torch.utils.data.DataLoader(clean_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)

    print(f"[4] Detecting on backdoor samples (n={len(bd_dataset)})")
    bd_results = detect_backdoor_samples_v2(model, bd_loader, args.device, args.num_levels)

    print(f"[5] Detecting on clean samples (n={len(clean_dataset)})")
    clean_results = detect_backdoor_samples_v2(model, clean_loader, args.device, args.num_levels)

    print("[6] Computing ROC metrics")

    # All individual indicators
    all_indicators = {
        # LP
        'lp_flip_count': 'LP Flip Count',
        'lp_conf_drop': 'LP Conf Drop',
        'lp_kl_div': 'LP KL Div',
        'lp_entropy_change': 'LP Entropy Change',
        # HP
        'hp_flip_count': 'HP Flip Count',
        'hp_conf_drop': 'HP Conf Drop',
        'hp_kl_div': 'HP KL Div',
        'hp_entropy_change': 'HP Entropy Change',
        # DCT
        'dct_flip_count': 'DCT Flip Count',
        'dct_conf_drop': 'DCT Conf Drop',
        'dct_kl_div': 'DCT KL Div',
        'dct_entropy_change': 'DCT Entropy Change',
        # Spatial
        's_flip_count': 'S Flip Count',
        's_conf_drop': 'S Conf Drop',
        's_kl_div': 'S KL Div',
        's_entropy_change': 'S Entropy Change',
    }

    print(f"\n{'='*60}")
    print(f"Per-Indicator Detection Results")
    print(f"{'='*60}")
    print(f"{'Indicator':<30} {'AUC':>8} {'F1':>8}")
    print(f"{'-'*60}")

    indicator_results = {}
    for key, name in all_indicators.items():
        bd_scores = bd_results[key]
        clean_scores = clean_results[key]
        combined_scores = np.concatenate([bd_scores, clean_scores])
        combined_labels = np.concatenate([np.ones(len(bd_scores)), np.zeros(len(clean_scores))])
        m = compute_roc_metrics(combined_scores, combined_labels)
        indicator_results[key] = m
        print(f"{name:<30} {m['roc_auc']:>8.4f} {m['best_f1']:>8.4f}")

    # ML Fusion with Cross-Validation (no data leakage)
    print(f"\n{'='*60}")
    print(f"ML Fusion Results (5-Fold Cross-Validation)")
    print(f"{'='*60}")

    feature_keys = list(all_indicators.keys())
    proba, clf, ml_metrics = train_ml_fusion(bd_results, clean_results, feature_keys, method='cv')
    print(f"Logistic Regression (All Features): AUC={ml_metrics['roc_auc']:.4f}, F1={ml_metrics['best_f1']:.4f}")

    # Best indicators only (top 5 by individual AUC)
    sorted_indicators = sorted(indicator_results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
    top_keys = [k for k, v in sorted_indicators[:5]]
    proba_top, _, ml_metrics_top = train_ml_fusion(bd_results, clean_results, top_keys, method='cv')
    print(f"Logistic Regression (Top 5): AUC={ml_metrics_top['roc_auc']:.4f}, F1={ml_metrics_top['best_f1']:.4f}")

    # Best single
    best_key = sorted_indicators[0][0]
    best_auc = sorted_indicators[0][1]['roc_auc']
    print(f"\nBest Single Indicator: {all_indicators[best_key]} (AUC={best_auc:.4f})")

    # Cross-validated ML fusion should be compared with best single
    if ml_metrics['roc_auc'] > best_auc:
        print(f">>> ML Fusion improves over best single by {ml_metrics['roc_auc'] - best_auc:.4f}")

    # Save
    print(f"\n[7] Saving to {save_path}/saved/teco_enhanced_v2/")
    os.makedirs(save_path + '/saved/teco_enhanced_v2', exist_ok=True)
    torch.save({
        'bd_results': bd_results,
        'clean_results': clean_results,
        'indicator_results': indicator_results,
        'ml_metrics': ml_metrics,
        'ml_metrics_top5': ml_metrics_top,
        'args': args.__dict__
    }, save_path + '/saved/teco_enhanced_v2/detection_results.pt')

    print("Done!")
