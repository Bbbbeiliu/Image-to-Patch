import argparse
import csv
import json
import os
from pathlib import Path
import sys

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--bins', type=int, default=256)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Choose a new output directory')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark'))
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from diagnostics import probability_histograms, summarize_histograms
    rows = list(csv.DictReader((a.evaluation / 'per_case_metrics.csv').open(encoding='utf-8-sig')))
    if not rows:
        raise ValueError('No cases in evaluation')
    foreground = np.zeros(a.bins, dtype=np.int64)
    background = np.zeros(a.bins, dtype=np.int64)

    def image(folder, sid):
        data = np.fromfile(a.evaluation / folder / (sid + '.png'), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    for row in rows:
        sid = row['sample_id']
        gt = image('ground_truth', sid) > 0
        path = a.evaluation / 'probabilities' / (sid + '.npy')
        if not path.is_file():
            path = a.evaluation / 'scores' / (sid + '.npy')
        if not path.is_file():
            raise FileNotFoundError('Continuous probability export required for ' + sid)
        probability = np.load(path)
        hist = probability_histograms(probability, gt, bins=a.bins)
        if isinstance(hist, dict):
            foreground += np.asarray(hist['foreground'], dtype=np.int64)
            background += np.asarray(hist['background'], dtype=np.int64)
        else:
            fg, bg = hist
            foreground += np.asarray(fg, dtype=np.int64)
            background += np.asarray(bg, dtype=np.int64)
    curves = summarize_histograms(foreground, background)
    a.out.mkdir(parents=True)
    (a.out / 'roc_pr.json').write_text(json.dumps(curves, indent=2), encoding='utf-8')
    counts = {k: sum((int(r[k]) for r in rows)) for k in ['tp', 'fp', 'fn', 'tn']}
    confusion = np.array([[counts['tp'], counts['fn']], [counts['fp'], counts['tn']]], dtype=np.int64)
    np.savetxt(a.out / 'confusion_counts.csv', confusion, delimiter=',', fmt='%d')
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].plot(curves['fpr'], curves['tpr'])
    axes[0].set(xlabel='False-positive rate', ylabel='True-positive rate', title='ROC')
    axes[1].plot(curves['recall'], curves['precision'])
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision–recall')
    fig.tight_layout()
    fig.savefig(a.out / 'roc_pr.png', dpi=200)
    plt.close(fig)
    normalized = confusion / confusion.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f'{confusion[row, col]:,}\n{normalized[row, col]:.2%}', ha='center', va='center', color='white' if normalized[row, col] > 0.5 else 'black')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Scar', 'Background'], yticklabels=['Scar', 'Background'], xlabel='Prediction', ylabel='Reference')
    fig.tight_layout()
    fig.savefig(a.out / 'confusion_matrix.png', dpi=200)
    plt.close(fig)
    selected = sorted(rows, key=lambda r: (float(r['Dice']), r['sample_id']))
    selected = selected[:3] + [selected[len(selected) // 2]]
    fig, axes = plt.subplots(len(selected), 4, figsize=(10, 2.7 * len(selected)))
    for axrow, row in zip(axes, selected):
        sid = row['sample_id']
        gt = image('ground_truth', sid) > 0
        pred = image('predictions', sid) > 0
        rgb = cv2.cvtColor(image('case_images', sid), cv2.COLOR_BGR2RGB)
        errors = np.zeros((*gt.shape, 3), dtype=np.uint8)
        errors[gt & pred] = (220, 220, 220)
        errors[gt & ~pred] = (40, 100, 230)
        errors[~gt & pred] = (220, 55, 55)
        for ax, value, title in zip(axrow, [rgb, gt, pred, errors], [sid, 'Reference', f"Prediction: Dice {float(row['Dice']):.3f}", 'Blue FN / red FP']):
            ax.imshow(value, cmap='gray')
            ax.set_title(title)
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(a.out / 'case_errors.png', dpi=200)
    plt.close(fig)
    print('Exported confusion counts, ROC/PR data and four diagnostic cases')
if __name__ == '__main__':
    main()
