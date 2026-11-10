import argparse
import csv
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='Directory of contour JSON files')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Choose a new output directory')
    import numpy as np
    from shapely.geometry import Polygon
    rows = []
    for f in sorted(a.input.glob('*.json')):
        source = json.loads(f.read_text(encoding='utf-8'))
        if 'points_mm' in source:
            points = np.asarray(source['points_mm'], dtype=float)
        else:
            points = np.asarray(source['nurbs_points'], dtype=float) * source['pixel_scale_mm_per_px']
        original = Polygon(points)
        if not original.is_valid:
            original = original.buffer(0)
        if original.is_empty or original.area <= 0:
            raise ValueError('Invalid reference contour: ' + f.name)
        for offset in [-3, -2, -1, 1, 2, 3]:
            patch = original.buffer(offset, join_style=1)
            shared = original.intersection(patch).area
            rows.append({'contour_id': f.stem, 'offset_mm': offset, 'reference_area_mm2': original.area, 'retained_reference_percent': 100 * shared / original.area, 'uncovered_percent': 100 * (original.area - shared) / original.area, 'extra_area_percent': 100 * patch.difference(original).area / original.area, 'empty_patch': int(patch.is_empty)})
    if not rows:
        raise ValueError('No contour JSON files found')
    a.out.mkdir(parents=True)
    with (a.out / 'per_contour.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for offset in [-3, -2, -1, 1, 2, 3]:
        subset = [r for r in rows if r['offset_mm'] == offset]
        row = {'offset_mm': offset, 'contours': len(subset), 'empty_patches': sum((r['empty_patch'] for r in subset))}
        for metric in ['retained_reference_percent', 'uncovered_percent', 'extra_area_percent']:
            values = [r[metric] for r in subset]
            row[metric + '_mean'] = float(np.mean(values))
            row[metric + '_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summaries.append(row)
    (a.out / 'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    print('Simulated', len(rows), 'uniform boundary offsets')
if __name__ == '__main__':
    main()
