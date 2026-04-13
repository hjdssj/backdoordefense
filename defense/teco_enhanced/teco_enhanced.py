"""
TeCo-Enhanced: Cross-Domain Robustness Consistency Detection

Extended from TeCo (CVPR 2023) to perform detection in both:
1. Spatial domain (image corruptions)
2. Frequency domain (FFT + low-pass filtering)

The key insight is that backdoor samples show inconsistent predictions
between spatial and frequency domain perturbations, while clean samples
remain consistent.

Basic structure:
    1. Load attack result (model + data)
    2. Spatial domain perturbation -> P_s trajectory
    3. Frequency domain perturbation -> P_f trajectory
    4. Compute cross-domain inconsistency: C = 1 - corr(P_s, P_f)
    5. Threshold-based detection
"""

import argparse
import os
import sys

sys.path.append('../')
sys.path.append(os.getcwd())

import numpy as np
import torch
# PyTorch 2.11+ changed torch.load default to weights_only=True,
# but our data contains PIL Images which are not allowed.
# Patch torch.load to default to weights_only=False for compatibility.
import functools
original_torch_load = torch.load
@functools.wraps(original_torch_load)
def patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# scipy >= 1.14 removed multichannel from gaussian_filter used by imagecorruptions
# Patch ndi.gaussian_filter to accept and ignore multichannel
import inspect
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

import yaml

import yaml
from PIL import Image
from scipy import stats
from sklearn import metrics
from sklearn.metrics import auc

from utils.aggregate_block.dataset_and_transform_generate import get_transform
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.save_load_attack import load_attack_result

from imagecorruptions import corrupt
from tqdm import tqdm


# ============================================================
# Spatial Domain Transforms (using imagecorruptions)
# ============================================================

# glass_blur is excluded due to scipy >= 1.14 incompatibility
SPATIAL_CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'motion_blur', 'zoom_blur',
    'snow', 'fog', 'brightness', 'contrast',
    'elastic_transform', 'pixelate', 'jpeg_compression'
]


def spatial_transform(image, corruption_name, severity):
    """
    Apply a spatial corruption to an image.

    Args:
        image: PIL Image or numpy array
        corruption_name: name of the corruption
        severity: 1-5

    Returns:
        corrupted PIL Image
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = corrupt(image, corruption_name=corruption_name, severity=severity)
    return Image.fromarray(image)


# ============================================================
# Frequency Domain Transforms (FFT + Low-pass filtering)
# ============================================================

def frequency_transform(image, cutoff_ratio=0.1, preserve_dc=True, filter_type='low'):
    """
    Apply frequency domain perturbation using FFT + filtering.

    Args:
        image: PIL Image or numpy array (H, W, C)
        cutoff_ratio: ratio of radius to image size (0-1), higher = more aggressive
        preserve_dc: if True, keep the DC component (mean color)
        filter_type: 'low' for low-pass, 'high' for high-pass

    Returns:
        filtered PIL Image
    """
    if isinstance(image, Image.Image):
        image = np.array(image).astype(np.float32)
    else:
        image = image.astype(np.float32)

    # Handle different image formats
    if len(image.shape) == 2:  # grayscale
        image = np.stack([image, image, image], axis=-1)

    h, w, c = image.shape

    # FFT for each channel
    filtered_channels = []
    for ch in range(c):
        # FFT
        f = np.fft.fft2(image[:, :, ch])
        fshift = np.fft.fftshift(f)

        # Create filter mask
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((y - center_y)**2 + (x - center_x)**2)

        # Cutoff radius based on ratio
        max_radius = np.sqrt(center_y**2 + center_x**2)
        cutoff_radius = cutoff_ratio * max_radius

        if filter_type == 'low':
            # Low-pass: keep high frequencies, remove low frequencies
            mask = (radius > cutoff_radius).astype(np.float32)
            if preserve_dc:
                mask[center_y, center_x] = 1.0
        else:
            # High-pass: keep low frequencies, remove high frequencies
            mask = (radius < cutoff_radius).astype(np.float32)
            if preserve_dc:
                mask[center_y, center_x] = 0.0

        # Apply mask
        fshift_filtered = fshift * mask

        # Inverse FFT
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.real(np.clip(img_back, 0, 255))

        filtered_channels.append(img_back)

    # Combine channels
    result = np.stack(filtered_channels, axis=-1).astype(np.uint8)
    return Image.fromarray(result)


def get_frequency_trajectory(image, model, device, num_levels=5, cutoff_range=(0.05, 0.5)):
    """
    Generate a trajectory of predictions under increasing frequency domain perturbations.

    Args:
        image: PIL Image (clean image)
        model: trained classifier
        device: cuda/cpu
        num_levels: number of severity levels
        cutoff_range: (min_cutoff, max_cutoff) - lower = more aggressive filtering

    Returns:
        list of prediction probabilities for each severity level
    """
    predictions = []
    min_cutoff, max_cutoff = cutoff_range

    for level in range(1, num_levels + 1):
        # Interpolate cutoff ratio (level 1 = most aggressive, level 5 = least)
        cutoff_ratio = max_cutoff - (max_cutoff - min_cutoff) * (level - 1) / (num_levels - 1)

        # Apply frequency transform
        perturbed_img = frequency_transform(image, cutoff_ratio=cutoff_ratio, preserve_dc=True)

        # Get prediction
        img_tensor = get_transform('cifar10', 32, 32)(perturbed_img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]

        predictions.append(probs.cpu().numpy())

    return predictions


def get_spatial_trajectory(image, model, device, num_levels=5):
    """
    Generate a trajectory of predictions under spatial corruptions.

    Args:
        image: PIL Image
        model: trained classifier
        device: cuda/cpu
        num_levels: number of severity levels

    Returns:
        list of prediction probabilities for each severity level per corruption,
        averaged across all corruption types
    """
    all_predictions = []

    for corruption in SPATIAL_CORRUPTIONS:
        corruption_preds = []
        for level in range(1, num_levels + 1):
            perturbed_img = spatial_transform(image, corruption, level)

            img_tensor = get_transform('cifar10', 32, 32)(perturbed_img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)[0]

            corruption_preds.append(probs.cpu().numpy())

        all_predictions.append(corruption_preds)

    # Average across all corruption types -> shape: (num_corruptions, num_levels, num_classes)
    # Then average across corruptions -> shape: (num_levels, num_classes)
    all_predictions = np.array(all_predictions)  # (15, 5, 10)
    avg_predictions = np.mean(all_predictions, axis=0)  # (5, 10)

    # Convert to list of arrays for consistency
    return [avg_predictions[level - 1] for level in range(1, num_levels + 1)]


def compute_cross_domain_consistency(P_s, P_f):
    """
    Compute cross-domain consistency C = 1 - correlation(P_s, P_f).

    For each domain we have a trajectory of predictions across severity levels.
    We compute the correlation between the spatial and frequency domain trajectories.

    Args:
        P_s: list of predicted probabilities for spatial domain, shape (num_levels, num_classes)
        P_f: list of predicted probabilities for frequency domain, shape (num_levels, num_classes)

    Returns:
        C: cross-domain inconsistency score (0 = perfect consistency, 1 = perfect inconsistency)
    """
    # Stack into arrays
    P_s = np.array(P_s)  # (num_levels, num_classes)
    P_f = np.array(P_f)  # (num_levels, num_classes)

    # Compute mean prediction per level, then correlate
    mean_ps = np.mean(P_s, axis=1)  # (num_levels,)
    mean_pf = np.mean(P_f, axis=1)  # (num_levels,)

    # Add small smoothing to avoid NaN when stddev is near zero
    eps = 1e-8
    std_ps = np.std(mean_ps) + eps
    std_pf = np.std(mean_pf) + eps

    # Safe correlation computation
    cov = np.mean((mean_ps - np.mean(mean_ps)) * (mean_pf - np.mean(mean_pf)))
    correlation = cov / (std_ps * std_pf)
    correlation = np.clip(correlation, -1.0, 1.0)

    C = 1 - correlation
    return C


def compute_flip_metrics(P_perturbed, P_orig, num_levels):
    """
    Compute flip-related metrics for a perturbation trajectory.

    Args:
        P_perturbed: predictions trajectory (num_levels, num_classes)
        P_orig: original prediction (num_classes,)
        num_levels: number of perturbation levels

    Returns:
        flip: whether prediction flipped at any level
        flip_count: number of levels where prediction flipped
        first_flip_level: first level where flip occurred (num_levels+1 if never flipped)
        conf_drop_first: confidence drop at first level
        conf_drop_avg: average confidence drop across all levels
    """
    P_orig = np.array(P_orig)
    P_perturbed = np.array(P_perturbed)

    orig_class = np.argmax(P_orig)
    orig_conf = P_orig[orig_class]

    # Class flips across levels
    perturbed_classes = np.argmax(P_perturbed, axis=1)  # (num_levels,)
    flips = perturbed_classes != orig_class

    flip = int(any(flips))
    flip_count = int(np.sum(flips))

    # First flip level (num_levels if never flipped)
    if any(flips):
        first_flip_level = int(np.argmax(flips) + 1)  # 1-indexed
    else:
        first_flip_level = num_levels + 1  # indicates never flipped

    # Confidence drop at first level
    perturbed_first = P_perturbed[0]
    if orig_class < len(perturbed_first):
        conf_drop_first = float(P_orig[orig_class] - perturbed_first[orig_class])
    else:
        conf_drop_first = 0.0

    # Average confidence drop across all levels
    perturbed_max = np.max(P_perturbed, axis=1)  # max prob at each level
    conf_drops = orig_conf - perturbed_max
    conf_drop_avg = float(np.mean(conf_drops))

    return flip, flip_count, first_flip_level, conf_drop_first, conf_drop_avg


def compute_confidence_drop(P_s, P_f, P_f_hp, P_orig, num_levels):
    """
    Compute confidence drop and class flip metrics for all domains.

    Args:
        P_s: spatial domain predictions trajectory
        P_f: low-pass frequency domain predictions trajectory
        P_f_hp: high-pass frequency domain predictions trajectory
        P_orig: original prediction probability
        num_levels: number of perturbation levels

    Returns:
        flip_s, flip_f, flip_f_hp: class flip indicators
        flip_count_s, flip_count_f, flip_count_f_hp: flip counts
        first_flip_s, first_flip_f, first_flip_f_hp: first flip level
        conf_drop_s, conf_drop_f, conf_drop_f_hp: first-level confidence drops
        conf_drop_avg_s, conf_drop_avg_f, conf_drop_avg_f_hp: avg confidence drops
    """
    flip_s, flip_count_s, first_flip_s, conf_drop_s, conf_drop_avg_s = compute_flip_metrics(P_s, P_orig, num_levels)
    flip_f, flip_count_f, first_flip_f, conf_drop_f, conf_drop_avg_f = compute_flip_metrics(P_f, P_orig, num_levels)
    flip_f_hp, flip_count_f_hp, first_flip_f_hp, conf_drop_f_hp, conf_drop_avg_f_hp = compute_flip_metrics(P_f_hp, P_orig, num_levels)

    return (flip_s, flip_f, flip_f_hp,
            flip_count_s, flip_count_f, flip_count_f_hp,
            first_flip_s, first_flip_f, first_flip_f_hp,
            conf_drop_s, conf_drop_f, conf_drop_f_hp,
            conf_drop_avg_s, conf_drop_avg_f, conf_drop_avg_f_hp)


def compute_domain_specific_metrics(P_s, P_f):
    """
    Compute domain-specific consistency metrics.

    Returns:
        C_s: spatial domain self-consistency (variance across severity)
        C_f: frequency domain self-consistency (variance across severity)
        C_cross: cross-domain consistency
    """
    P_s = np.array(P_s)
    P_f = np.array(P_f)

    # Self-consistency: how stable is the prediction across severity levels
    # Lower = more consistent
    mean_ps = np.mean(P_s, axis=1)
    mean_pf = np.mean(P_f, axis=1)

    # Variance of mean predictions across levels
    C_s = np.std(mean_ps)  # spatial self-consistency
    C_f = np.std(mean_pf)   # frequency self-consistency

    # Cross-domain consistency
    C_cross = compute_cross_domain_consistency(P_s, P_f)

    return C_s, C_f, C_cross


# ============================================================
# Main Detection Function
# ============================================================

def detect_backdoor_samples(model, data_loader, device, num_levels=5):
    """
    Detect backdoor samples using cross-domain robustness consistency.

    Args:
        model: trained backdoor-infected classifier
        data_loader: data loader containing samples to test
        device: cuda/cpu
        num_levels: number of perturbation severity levels

    Returns:
        results: dict with metrics for each sample
    """
    results = {
        'indices': [],
        'C_s': [],         # spatial self-consistency
        'C_f': [],         # low-pass frequency self-consistency
        'C_f_hp': [],      # high-pass frequency self-consistency
        'C_cross': [],     # cross-domain consistency
        'flip_s': [],
        'flip_f': [],      # low-pass flip
        'flip_f_hp': [],   # high-pass flip
        'flip_count_s': [],
        'flip_count_f': [],   # low-pass flip count
        'flip_count_f_hp': [],  # high-pass flip count
        'first_flip_s': [],
        'first_flip_f': [],   # low-pass: first flip level
        'first_flip_f_hp': [],  # high-pass: first flip level
        'conf_drop_s': [],
        'conf_drop_f': [],   # low-pass confidence drop
        'conf_drop_f_hp': [],  # high-pass confidence drop
        'conf_drop_avg_s': [],
        'conf_drop_avg_f': [],
        'conf_drop_avg_f_hp': [],
        'predictions': []
    }

    model.eval()

    for idx, (inputs, labels) in enumerate(tqdm(data_loader, desc="Detecting")):
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            original_preds = probs.cpu().numpy()

        # Process each sample in batch
        for i in range(inputs.size(0)):
            sample_idx = idx * data_loader.batch_size + i
            img = inputs[i].cpu()

            # Convert tensor to PIL Image for transforms
            # CIFAR-10 normalization: mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
            img_unnorm = img * 0.5 + 0.5
            img_unnorm = torch.clamp(img_unnorm, 0, 1)
            img_pil = Image.fromarray((img_unnorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

            # Get original prediction
            P_orig = original_preds[i]

            # Get spatial domain trajectory
            P_s = []
            for level in range(1, num_levels + 1):
                level_preds = []
                for corruption in SPATIAL_CORRUPTIONS[:5]:
                    perturbed = spatial_transform(img_pil, corruption, level)
                    img_t = get_transform('cifar10', 32, 32)(perturbed).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out = model(img_t)
                        prob = torch.softmax(out, dim=1)[0].cpu().numpy()
                    level_preds.append(prob)
                P_s.append(np.mean(level_preds, axis=0))

            # Get low-pass frequency domain trajectory
            P_f = []
            lp_cutoff_range = (0.05, 0.5)
            min_cutoff, max_cutoff = lp_cutoff_range
            for level in range(1, num_levels + 1):
                cutoff_ratio = max_cutoff - (max_cutoff - min_cutoff) * (level - 1) / (num_levels - 1)
                perturbed = frequency_transform(img_pil, cutoff_ratio=cutoff_ratio, filter_type='low')
                img_t = get_transform('cifar10', 32, 32)(perturbed).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(img_t)
                    prob = torch.softmax(out, dim=1)[0].cpu().numpy()
                P_f.append(prob)

            # Get high-pass frequency domain trajectory
            P_f_hp = []
            hp_cutoff_range = (0.05, 0.5)
            min_cutoff, max_cutoff = hp_cutoff_range
            for level in range(1, num_levels + 1):
                cutoff_ratio = max_cutoff - (max_cutoff - min_cutoff) * (level - 1) / (num_levels - 1)
                perturbed = frequency_transform(img_pil, cutoff_ratio=cutoff_ratio, filter_type='high')
                img_t = get_transform('cifar10', 32, 32)(perturbed).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(img_t)
                    prob = torch.softmax(out, dim=1)[0].cpu().numpy()
                P_f_hp.append(prob)

            # Compute metrics
            P_f_all = np.array(P_f)
            P_f_hp_all = np.array(P_f_hp)
            mean_pf = np.mean(P_f_all, axis=1)
            mean_pf_hp = np.mean(P_f_hp_all, axis=1)

            C_s = float(np.std(np.mean(np.array(P_s), axis=1)))
            C_f = float(np.std(mean_pf))
            C_f_hp = float(np.std(mean_pf_hp))
            C_cross = compute_cross_domain_consistency(P_s, P_f)

            (flip_s, flip_f, flip_f_hp,
             flip_count_s, flip_count_f, flip_count_f_hp,
             first_flip_s, first_flip_f, first_flip_f_hp,
             conf_drop_s, conf_drop_f, conf_drop_f_hp,
             conf_drop_avg_s, conf_drop_avg_f, conf_drop_avg_f_hp) = compute_confidence_drop(P_s, P_f, P_f_hp, P_orig, num_levels)

            results['indices'].append(sample_idx)
            results['C_s'].append(C_s)
            results['C_f'].append(C_f)
            results['C_f_hp'].append(C_f_hp)
            results['C_cross'].append(C_cross)
            results['flip_s'].append(flip_s)
            results['flip_f'].append(flip_f)
            results['flip_f_hp'].append(flip_f_hp)
            results['flip_count_s'].append(flip_count_s)
            results['flip_count_f'].append(flip_count_f)
            results['flip_count_f_hp'].append(flip_count_f_hp)
            results['first_flip_s'].append(first_flip_s)
            results['first_flip_f'].append(first_flip_f)
            results['first_flip_f_hp'].append(first_flip_f_hp)
            results['conf_drop_s'].append(conf_drop_s)
            results['conf_drop_f'].append(conf_drop_f)
            results['conf_drop_f_hp'].append(conf_drop_f_hp)
            results['conf_drop_avg_s'].append(conf_drop_avg_s)
            results['conf_drop_avg_f'].append(conf_drop_avg_f)
            results['conf_drop_avg_f_hp'].append(conf_drop_avg_f_hp)
            results['predictions'].append(P_orig)

    # Convert to arrays
    for key in ['C_s', 'C_f', 'C_f_hp', 'C_cross', 'flip_s', 'flip_f', 'flip_f_hp',
                'flip_count_s', 'flip_count_f', 'flip_count_f_hp',
                'first_flip_s', 'first_flip_f', 'first_flip_f_hp',
                'conf_drop_s', 'conf_drop_f', 'conf_drop_f_hp',
                'conf_drop_avg_s', 'conf_drop_avg_f', 'conf_drop_avg_f_hp']:
        results[key] = np.array(results[key])
    results['predictions'] = np.array(results['predictions'])

    return results


def compute_roc_metrics(scores, labels):
    """
    Compute ROC-AUC and F1 metrics for detection.

    Args:
        scores: detection scores (higher = more likely backdoor)
        labels: ground truth labels (1 = backdoor, 0 = clean)

    Returns:
        dict with fpr, tpr, thresholds, roc_auc, best_f1, best_threshold
    """
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Find best threshold based on F1
    f1_scores = []
    for th in thresholds:
        pred = (scores > th).astype(int)
        f1 = metrics.f1_score(labels, pred)
        f1_scores.append(f1)

    f1_scores = np.array(f1_scores)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_threshold = thresholds[best_idx]

    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'roc_auc': roc_auc,
        'best_f1': best_f1,
        'best_threshold': best_threshold
    }


# ============================================================
# CLI Interface
# ============================================================

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--result_file', type=str, required=True, help='attack result folder name')
    parser.add_argument('--yaml_path', type=str, default='./config/defense/teco_enhanced/cifar10.yaml')
    parser.add_argument('--num_levels', type=int, default=5, help='number of perturbation severity levels')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Load config
    with open(args.yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    config.update({k: v for k, v in args.__dict__.items() if v is not None})
    args.__dict__ = config

    # Dataset settings
    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == 'gtsrb':
        args.num_classes = 43
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == 'tiny':
        args.num_classes = 200
        args.input_height = 64
        args.input_width = 64
        args.input_channel = 3

    save_path = 'record/' + args.result_file

    print(f"[1] Loading attack result from {save_path}/attack_result.pt")
    result = load_attack_result(save_path + '/attack_result.pt')

    # Load model
    print("[2] Loading model")
    model = generate_cls_model(args.model or 'preactresnet18', args.num_classes)
    model.load_state_dict(result['model'])
    model.to(args.device)
    model.eval()

    # Prepare data
    print("[3] Preparing data")
    tran = get_transform(args.dataset, args.input_height, args.input_width, train=False)

    # Backdoor test data
    x_bd = result['bd_test']['x']
    y_bd = result['bd_test']['y']
    data_bd = list(zip(x_bd, y_bd))
    bd_dataset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_bd,
        poison_idx=np.zeros(len(data_bd)),
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    bd_loader = torch.utils.data.DataLoader(
        bd_dataset, batch_size=args.batch_size, num_workers=0,
        shuffle=False, pin_memory=False
    )

    # Clean test data
    x_clean = result['clean_test']['x']
    y_clean = result['clean_test']['y']
    data_clean = list(zip(x_clean, y_clean))
    clean_dataset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_clean,
        poison_idx=np.zeros(len(data_clean)),
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    clean_loader = torch.utils.data.DataLoader(
        clean_dataset, batch_size=args.batch_size, num_workers=0,
        shuffle=False, pin_memory=False
    )

    # Detect on backdoor samples
    print(f"[4] Detecting on backdoor samples (n={len(bd_dataset)})")
    bd_results = detect_backdoor_samples(model, bd_loader, args.device, args.num_levels)

    # Detect on clean samples
    print(f"[5] Detecting on clean samples (n={len(clean_dataset)})")
    clean_results = detect_backdoor_samples(model, clean_loader, args.device, args.num_levels)

    # Combine results
    print("[6] Computing ROC metrics")

    # All available metrics
    all_metrics = {
        # Self-consistency metrics
        'C_s (Spatial Self-Consistency)': 'C_s',
        'C_f (Low-Pass Self-Consistency)': 'C_f',
        'C_f_hp (High-Pass Self-Consistency)': 'C_f_hp',
        'C_cross (Cross-Domain Consistency)': 'C_cross',
        # Flip rate metrics
        'flip_s (Spatial Flip Rate)': 'flip_s',
        'flip_f (Low-Pass Flip Rate)': 'flip_f',
        'flip_f_hp (High-Pass Flip Rate)': 'flip_f_hp',
        # Flip count metrics
        'flip_count_s (Spatial Flip Count)': 'flip_count_s',
        'flip_count_f (Low-Pass Flip Count)': 'flip_count_f',
        'flip_count_f_hp (High-Pass Flip Count)': 'flip_count_f_hp',
        # First flip level (higher = more backdoor, i.e., flips early)
        'first_flip_f (Low-Pass First Flip Level)': 'first_flip_f',
        'first_flip_f_hp (High-Pass First Flip Level)': 'first_flip_f_hp',
        # Confidence drop metrics
        'conf_drop_s (Spatial Conf Drop)': 'conf_drop_s',
        'conf_drop_f (Low-Pass Conf Drop)': 'conf_drop_f',
        'conf_drop_f_hp (High-Pass Conf Drop)': 'conf_drop_f_hp',
        'conf_drop_avg_f (Low-Pass Avg Conf Drop)': 'conf_drop_avg_f',
        'conf_drop_avg_f_hp (High-Pass Avg Conf Drop)': 'conf_drop_avg_f_hp',
    }

    # Compute metrics for each indicator
    print(f"\n{'='*60}")
    print(f"Per-Indicator Detection Results")
    print(f"{'='*60}")
    print(f"{'Indicator':<40} {'AUC':>8} {'F1':>8} {'Threshold':>12}")
    print(f"{'-'*60}")

    indicator_results = {}
    for name, key in all_metrics.items():
        bd_scores = bd_results[key]
        clean_scores = clean_results[key]
        combined_scores = np.concatenate([bd_scores, clean_scores])
        combined_labels = np.concatenate([np.ones(len(bd_scores)), np.zeros(len(clean_scores))])
        m = compute_roc_metrics(combined_scores, combined_labels)
        indicator_results[key] = m
        print(f"{name:<40} {m['roc_auc']:>8.4f} {m['best_f1']:>8.4f} {m['best_threshold']:>12.6f}")

    # Combined score: weighted average of normalized scores
    # Higher is more likely backdoor for all metrics

    # Z-score normalize each metric, then combine
    def normalize_scores(bd_scores, clean_scores):
        all_scores = np.concatenate([bd_scores, clean_scores])
        mean = np.mean(all_scores)
        std = np.std(all_scores) + 1e-8
        return (all_scores - mean) / std

    # Combined score: best indicators from low-pass + high-pass
    # Use: flip_count_f, conf_drop_avg_f (low-pass), flip_count_f_hp, conf_drop_avg_f_hp (high-pass)
    combined = (
        normalize_scores(bd_results['flip_count_f'], clean_results['flip_count_f']) +
        normalize_scores(bd_results['conf_drop_avg_f'], clean_results['conf_drop_avg_f']) +
        normalize_scores(bd_results['flip_count_f_hp'], clean_results['flip_count_f_hp']) +
        normalize_scores(bd_results['conf_drop_avg_f_hp'], clean_results['conf_drop_avg_f_hp']) +
        normalize_scores(bd_results['first_flip_f'], clean_results['first_flip_f'])
    ) / 5.0

    bd_labels = np.concatenate([np.ones(len(bd_results['flip_count_f'])), np.zeros(len(clean_results['flip_count_f']))])
    combined_metrics = compute_roc_metrics(combined, bd_labels)

    print(f"\n{'='*60}")
    print(f"Combined Score (5-Indicator LP+HP)")
    print(f"{'='*60}")
    print(f"ROC-AUC: {combined_metrics['roc_auc']:.4f}")
    print(f"Best F1: {combined_metrics['best_f1']:.4f}")
    print(f"Best Threshold: {combined_metrics['best_threshold']:.6f}")

    # Choose best performing indicator
    best_key = max(indicator_results.keys(), key=lambda k: indicator_results[k]['roc_auc'])
    best_auc = indicator_results[best_key]['roc_auc']
    print(f"\nBest Single Indicator: {best_key} (AUC={best_auc:.4f})")
    if combined_metrics['roc_auc'] > best_auc:
        print(f">>> Combined score outperforms best single indicator!")

    # Save results
    print(f"\n[7] Saving results to {save_path}/saved/teco_enhanced/")
    os.makedirs(save_path + '/saved/teco_enhanced', exist_ok=True)

    save_dict = {
        'bd_results': bd_results,
        'clean_results': clean_results,
        'indicator_results': indicator_results,
        'combined_metrics': combined_metrics,
        'best_indicator': best_key,
        'args': args.__dict__
    }

    torch.save(save_dict, save_path + '/saved/teco_enhanced/detection_results.pt')

    print("Done!")
