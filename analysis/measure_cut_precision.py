import argparse
import csv
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--scans', type=Path, required=True)
    p.add_argument('--pattern', default='scan_*.png')
    p.add_argument('--dpi', type=float, default=200)
    p.add_argument('--ddof', type=int, choices=[0, 1], default=0)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Choose a new output directory')
    import numpy as np
    from cut_precision_core import load_plan, extract_largest_contour, compute_metrics_from_pixels, align_contour_by_farthest_axis, align_contour_with_best_flip
    area, width, height, scale, points = load_plan(str(a.plan))
    aligned = align_contour_by_farthest_axis(points)
    rows = []
    for scan in sorted(a.scans.glob(a.pattern)):
        contour = extract_largest_contour(str(scan))
        measured_area, measured_width, measured_height = compute_metrics_from_pixels(contour, 25.4 / a.dpi)
        _, flip, deviation = align_contour_with_best_flip(contour * (25.4 / a.dpi), aligned)
        rows.append({'sample': scan.stem, 'actual_area_mm2': measured_area, 'actual_width_mm': measured_width, 'actual_height_mm': measured_height, 'mean_deviation_mm': deviation, 'best_flip': flip})
    if not rows or len(rows) <= a.ddof:
        raise ValueError('Insufficient scans')
    a.out.mkdir(parents=True)
    with (a.out / 'per_cut.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'n': len(rows), 'dpi': a.dpi, 'sd_ddof': a.ddof}
    for metric in ['actual_area_mm2', 'actual_width_mm', 'actual_height_mm', 'mean_deviation_mm']:
        summary[metric] = {'mean': float(np.mean([r[metric] for r in rows])), 'sd': float(np.std([r[metric] for r in rows], ddof=a.ddof))}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
if __name__ == '__main__':
    main()
