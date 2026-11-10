import json
import cv2
import numpy as np
import os
import glob
import csv
import matplotlib.pyplot as plt
PLAN_JSON = 'contour_001.json'
SCAN_DIR = 'scans'
SCAN_PATTERN = 'scan_*.png'
DPI = 200
OUTPUT_CSV = 'repeat_accuracy_detailed.csv'
SAVE_ALIGNED_PLOTS = True

def load_plan(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    width_cm = data.get('actual_width_cm', 0)
    height_cm = data.get('actual_height_cm', 0)
    plan_width_mm = max(width_cm, height_cm) * 10.0
    plan_height_mm = min(width_cm, height_cm) * 10.0
    points_px = np.array(data['nurbs_points'], dtype=np.float32)
    scale_mm_per_px = data['pixel_scale_mm_per_px']
    plan_mm = points_px * scale_mm_per_px
    plan_contour = plan_mm.reshape(-1, 1, 2).astype(np.float32)
    plan_area_mm2 = cv2.contourArea(plan_contour)
    return (plan_area_mm2, plan_width_mm, plan_height_mm, scale_mm_per_px, plan_mm)

def extract_largest_contour(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {image_path}')
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError(f'No contour found: {image_path}')
    largest = max(contours, key=cv2.contourArea)
    return largest.squeeze(1)

def get_farthest_points(contour):
    max_dist = 0
    idx_a, idx_b = (0, 0)
    n = len(contour)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(contour[i] - contour[j])
            if d > max_dist:
                max_dist = d
                idx_a, idx_b = (i, j)
    return (contour[idx_a], contour[idx_b], max_dist)

def align_contour_by_farthest_axis(contour):
    pA, pB, _ = get_farthest_points(contour)
    C = (pA + pB) / 2.0
    vec = pB - pA
    angle = np.arctan2(vec[1], vec[0])
    cos_theta = np.cos(-angle)
    sin_theta = np.sin(-angle)
    R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    translated = contour - C
    rotated = translated @ R.T
    return rotated

def apply_flip(contour, flip_type):
    if flip_type == 'none':
        return contour
    elif flip_type == 'lr':
        return contour * np.array([-1, 1])
    elif flip_type == 'ud':
        return contour * np.array([1, -1])
    elif flip_type == 'both':
        return contour * np.array([-1, -1])
    else:
        raise ValueError('Unknown flip type')

def compute_vertical_deviation(plan_pts, actual_pts, num_samples=100):
    x_min = max(plan_pts[:, 0].min(), actual_pts[:, 0].min())
    x_max = min(plan_pts[:, 0].max(), actual_pts[:, 0].max())
    if x_min >= x_max:
        return (np.nan, np.nan)
    xs = np.linspace(x_min, x_max, num_samples)
    deviations = []
    for x in xs:
        plan_mask = np.abs(plan_pts[:, 0] - x) < (x_max - x_min) / num_samples * 1.5
        if not np.any(plan_mask):
            continue
        plan_ys = plan_pts[plan_mask, 1]
        plan_y_top = plan_ys.max()
        plan_y_bottom = plan_ys.min()
        actual_mask = np.abs(actual_pts[:, 0] - x) < (x_max - x_min) / num_samples * 1.5
        if not np.any(actual_mask):
            continue
        actual_ys = actual_pts[actual_mask, 1]
        actual_y_top = actual_ys.max()
        actual_y_bottom = actual_ys.min()
        dev_top = abs(actual_y_top - plan_y_top)
        dev_bottom = abs(actual_y_bottom - plan_y_bottom)
        deviations.append((dev_top + dev_bottom) / 2.0)
    if not deviations:
        return (np.nan, np.nan)
    return (np.mean(deviations), np.max(deviations))

def align_contour_with_best_flip(contour, target_contour):
    base_aligned = align_contour_by_farthest_axis(contour)
    best_dev = float('inf')
    best_aligned = None
    best_flip = None
    for flip_type in ['none', 'lr', 'ud', 'both']:
        flipped = apply_flip(base_aligned, flip_type)
        mean_dev, _ = compute_vertical_deviation(target_contour, flipped)
        if not np.isnan(mean_dev) and mean_dev < best_dev:
            best_dev = mean_dev
            best_aligned = flipped
            best_flip = flip_type
    if best_aligned is None:
        best_aligned = base_aligned
    return (best_aligned, best_flip, best_dev)

def compute_metrics_from_pixels(contour_px, mm_per_pixel):
    contour_mm = contour_px * mm_per_pixel
    contour_mm_reshaped = contour_mm.reshape(-1, 1, 2).astype(np.float32)
    area_mm2 = cv2.contourArea(contour_mm_reshaped)
    x, y, w_px, h_px = cv2.boundingRect(contour_px)
    w_mm = w_px * mm_per_pixel
    h_mm = h_px * mm_per_pixel
    width_mm = max(w_mm, h_mm)
    height_mm = min(w_mm, h_mm)
    return (area_mm2, width_mm, height_mm)
