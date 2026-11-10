from __future__ import annotations
import math
import numpy as np
from scipy import ndimage
_BOUNDARY_STRUCTURE = np.ones((3, 3), dtype=bool)

def _validated_mask(mask: np.ndarray, name: str, *, two_dimensional: bool=False) -> np.ndarray:
    array = np.asarray(mask)
    if array.dtype.kind != 'b':
        raise ValueError(f'{name} must be a boolean array')
    if two_dimensional and array.ndim != 2:
        raise ValueError(f'{name} must be a 2D per-case array')
    if array.size == 0:
        raise ValueError(f'{name} must not be empty')
    return array

def _validated_prob(prob: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(prob)
    if array.dtype.kind not in 'fiu':
        raise ValueError('prob must be a real numeric array')
    if array.shape != shape:
        raise ValueError(f'prob shape {array.shape} does not match mask shape {shape}')
    if array.size == 0:
        raise ValueError('prob must not be empty')
    if not np.all(np.isfinite(array)):
        raise ValueError('prob must contain only finite values')
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError('prob values must be within 0..1')
    return array

def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else math.nan

def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(mask, structure=_BOUNDARY_STRUCTURE, border_value=0)
    return mask & ~eroded

def case_diagnostics(pred: np.ndarray, gt: np.ndarray, prob: np.ndarray) -> dict[str, object]:
    pred = _validated_mask(pred, 'pred', two_dimensional=True)
    gt = _validated_mask(gt, 'gt', two_dimensional=True)
    if pred.shape != gt.shape:
        raise ValueError(f'pred shape {pred.shape} does not match gt shape {gt.shape}')
    prob = _validated_prob(prob, pred.shape)
    total = int(pred.size)
    pred_positive = int(np.count_nonzero(pred))
    gt_positive = int(np.count_nonzero(gt))
    tp = int(np.count_nonzero(pred & gt))
    fp = pred_positive - tp
    fn = gt_positive - tp
    tn = total - tp - fp - fn
    bg_tp, bg_fp, bg_fn = (tn, fn, fp)
    bg_dice_denominator = 2 * bg_tp + bg_fp + bg_fn
    bg_iou_denominator = bg_tp + bg_fp + bg_fn
    bg_precision_denominator = bg_tp + bg_fp
    bg_recall_denominator = bg_tp + bg_fn
    pred_empty = pred_positive == 0
    gt_empty = gt_positive == 0
    predicted_components = int(ndimage.label(pred, structure=_BOUNDARY_STRUCTURE)[1])
    reference_components = int(ndimage.label(gt, structure=_BOUNDARY_STRUCTURE)[1])
    if pred_empty and gt_empty:
        boundary_status = 'both_empty'
    elif pred_empty:
        boundary_status = 'empty_prediction'
    elif gt_empty:
        boundary_status = 'empty_reference'
    else:
        boundary_status = 'ok'
    boundary_precision = math.nan
    boundary_recall = math.nan
    boundary_f1 = math.nan
    hd95 = math.nan
    assd = math.nan
    if not pred_empty and (not gt_empty):
        pred_boundary = _mask_boundary(pred)
        gt_boundary = _mask_boundary(gt)
        distance_to_gt = ndimage.distance_transform_edt(~gt_boundary)
        distance_to_pred = ndimage.distance_transform_edt(~pred_boundary)
        pred_to_gt = distance_to_gt[pred_boundary]
        gt_to_pred = distance_to_pred[gt_boundary]
        boundary_precision = float(np.count_nonzero(pred_to_gt <= 1.0) / pred_to_gt.size)
        boundary_recall = float(np.count_nonzero(gt_to_pred <= 1.0) / gt_to_pred.size)
        if boundary_precision + boundary_recall:
            boundary_f1 = float(2.0 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall))
        else:
            boundary_f1 = 0.0
        hd95 = float(max(np.percentile(pred_to_gt, 95), np.percentile(gt_to_pred, 95)))
        assd = float((np.mean(pred_to_gt) + np.mean(gt_to_pred)) / 2.0)
    return {'foreground_fraction': float(gt_positive / total), 'predicted_fraction': float(pred_positive / total), 'foreground_components_8_connected': reference_components, 'predicted_components_8_connected': predicted_components, 'background_Dice': _ratio(2 * bg_tp, bg_dice_denominator), 'background_IoU': _ratio(bg_tp, bg_iou_denominator), 'background_Precision': _ratio(bg_tp, bg_precision_denominator), 'background_Recall': _ratio(bg_tp, bg_recall_denominator), 'specificity': _ratio(tn, tn + fp), 'accuracy': float((tp + tn) / total), 'prediction_empty': bool(pred_empty), 'reference_empty': bool(gt_empty), 'boundary_status': boundary_status, 'boundary_precision_1px': boundary_precision, 'boundary_recall_1px': boundary_recall, 'BoundaryF1_1px': boundary_f1, 'HD95_network_px': hd95, 'ASSD_network_px': assd}

def probability_histograms(prob: np.ndarray, gt: np.ndarray, bins: int=4096) -> tuple[np.ndarray, np.ndarray]:
    gt = _validated_mask(gt, 'gt')
    prob = _validated_prob(prob, gt.shape)
    if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)) or bins <= 0:
        raise ValueError('bins must be a positive integer')
    bins = int(bins)
    foreground, _ = np.histogram(prob[gt], bins=bins, range=(0.0, 1.0))
    background, _ = np.histogram(prob[~gt], bins=bins, range=(0.0, 1.0))
    return (foreground.astype(np.uint64, copy=False), background.astype(np.uint64, copy=False))

def _validated_counts(values: np.ndarray, name: str) -> list[int]:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f'{name} must be a non-empty 1D histogram')
    counts: list[int] = []
    for value in array.tolist():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f'{name} must contain integer counts')
        count = int(value)
        if count < 0:
            raise ValueError(f'{name} must contain nonnegative counts')
        counts.append(count)
    return counts

def summarize_histograms(fg: np.ndarray, bg: np.ndarray) -> dict[str, object]:
    fg_counts = _validated_counts(fg, 'fg')
    bg_counts = _validated_counts(bg, 'bg')
    if len(fg_counts) != len(bg_counts):
        raise ValueError('fg and bg histograms must have the same number of bins')
    n_bins = len(fg_counts)
    fg_total = sum(fg_counts)
    bg_total = sum(bg_counts)
    total = fg_total + bg_total

    def suffix_sums(counts: list[int]) -> list[int]:
        result = [0] * (len(counts) + 1)
        for index in range(len(counts) - 1, -1, -1):
            result[index] = result[index + 1] + counts[index]
        return result
    fg_suffix = suffix_sums(fg_counts)
    bg_suffix = suffix_sums(bg_counts)
    threshold_indices = list(range(n_bins, -1, -1))
    thresholds = [float(index / n_bins) for index in threshold_indices]
    tp = [int(fg_suffix[index]) for index in threshold_indices]
    fp = [int(bg_suffix[index]) for index in threshold_indices]
    fn = [int(fg_total - value) for value in tp]
    tn = [int(bg_total - value) for value in fp]
    precision = [float(true_positive / (true_positive + false_positive)) if true_positive + false_positive else 1.0 for true_positive, false_positive in zip(tp, fp)]
    recall: list[float | None] = [float(value / fg_total) if fg_total else None for value in tp]
    fpr: list[float | None] = [float(value / bg_total) if bg_total else None for value in fp]
    specificity: list[float | None] = [float(value / bg_total) if bg_total else None for value in tn]
    accuracy: list[float | None] = [float((true_positive + true_negative) / total) if total else None for true_positive, true_negative in zip(tp, tn)]
    dice: list[float | None] = []
    iou: list[float | None] = []
    for true_positive, false_positive, false_negative in zip(tp, fp, fn):
        dice_denominator = 2 * true_positive + false_positive + false_negative
        iou_denominator = true_positive + false_positive + false_negative
        dice.append(float(2 * true_positive / dice_denominator) if dice_denominator else None)
        iou.append(float(true_positive / iou_denominator) if iou_denominator else None)
    roc_auc: float | None = None
    if fg_total and bg_total:
        roc_auc = float(sum(((fpr[index] - fpr[index - 1]) * (recall[index] + recall[index - 1]) / 2.0 for index in range(1, len(thresholds)))))
    average_precision: float | None = None
    if fg_total:
        average_precision = float(sum(((recall[index] - recall[index - 1]) * precision[index] for index in range(1, len(thresholds)))))
    return {'n_bins': int(n_bins), 'foreground_pixels': int(fg_total), 'background_pixels': int(bg_total), 'total_pixels': int(total), 'thresholds': thresholds, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'fpr': fpr, 'tpr': recall.copy(), 'precision': precision, 'recall': recall, 'specificity': specificity, 'accuracy': accuracy, 'dice': dice, 'iou': iou, 'roc_auc_trapezoid': roc_auc, 'average_precision_stepwise': average_precision, 'curve_method': 'Approximate pooled-pixel curves at histogram-bin boundaries. The threshold 1.0 row is the zero-predicted-positive anchor because the histogram cannot isolate exact probabilities of 1.0.', 'average_precision_method': 'Stepwise recall integration using precision after each added bin; precision is 1.0 at the zero-predicted-positive baseline. This estimate depends on histogram resolution and pooled pixels.'}
