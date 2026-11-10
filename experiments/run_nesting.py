import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', type=Path, required=True, help='Parent of Dataset_40, Dataset_80, Dataset_100 and Dataset_130')
    p.add_argument('--sizes', type=int, nargs='+', default=[40, 80, 100, 130])
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Choose a new output directory')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fabrication/nesting'))
    import comparison_methods as m
    import numpy as np
    m.tqdm = lambda sequence, **kwargs: sequence
    methods = {'PolarGreedy': m.polar_greedy_nesting, 'RectPack': m.rect_pack_nesting, 'NFPPlacer': m.nfp_nesting}
    a.out.mkdir(parents=True)
    rows = []
    for size in a.sizes:
        for repeat in range(a.repeats + 1):
            order = list(methods)
            random.Random(20261105 + size * 100 + repeat).shuffle(order)
            for name in order:
                contours = m.load_dataset(str(a.datasets / f'Dataset_{size}'))
                if len(contours) != size:
                    raise ValueError('Unexpected contour count')
                m.NFP_CACHE.clear()
                start = time.perf_counter()
                pages = methods[name](contours, m.MARGIN_PX)
                elapsed = time.perf_counter() - start
                if repeat:
                    rows.append({'dataset': size, 'method': name, 'repeat': repeat, 'time_s': elapsed, 'pages': len(pages), 'utilization': m.compute_avg_util(pages), 'placed': sum((len(page) for page in pages))})
                print(size, name, repeat, round(elapsed, 4), flush=True)
    with (a.out / 'raw.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for size in a.sizes:
        for name in methods:
            subset = [r for r in rows if r['dataset'] == size and r['method'] == name]
            row = {'dataset': size, 'method': name, 'n': len(subset)}
            for metric in ['time_s', 'utilization', 'pages']:
                row[metric + '_mean'] = float(np.mean([r[metric] for r in subset]))
                row[metric + '_sd'] = float(np.std([r[metric] for r in subset], ddof=1))
            summaries.append(row)
    (a.out / 'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
if __name__ == '__main__':
    main()
