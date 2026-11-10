import argparse
import json
import os
from pathlib import Path
import sys

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--masks', type=Path, required=True)
    p.add_argument('--sample-id', required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--grid', type=int, default=4)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Choose a new output directory')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark'))
    import benchmark as b
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    build, _, dataset, _, transform, *_ = b.project_modules()
    ds = dataset(str(a.images), str(a.masks), transform=transform)
    index = [Path(name).stem for name in ds.image_names].index(a.sample_id)
    x, gt, _ = ds[index]
    x = x[None].to(a.device)
    target = gt[None].to(a.device) > 0.5
    if not target.any():
        raise ValueError('Annotated foreground target is empty')
    model = build('retained').to(a.device).eval()
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    state = saved.get('state_dict', saved.get('model_state', saved))
    model.load_state_dict(state, strict=True)
    h, w = x.shape[-2:]
    means = x.mean(dim=(2, 3), keepdim=True)
    drops = np.zeros((a.grid, a.grid), dtype=float)
    with torch.inference_mode():
        baseline = float(model(x).sigmoid()[target].mean().item())
        for row in range(a.grid):
            for col in range(a.grid):
                y0, y1 = (row * h // a.grid, (row + 1) * h // a.grid)
                x0, x1 = (col * w // a.grid, (col + 1) * w // a.grid)
                masked = x.clone()
                masked[:, :, y0:y1, x0:x1] = means
                score = float(model(masked).sigmoid()[target].mean().item())
                drops[row, col] = baseline - score
    a.out.mkdir(parents=True)
    (a.out / 'occlusion.json').write_text(json.dumps({'baseline_target_probability': baseline, 'grid': a.grid, 'probability_drop': drops.tolist()}, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(5, 4))
    plot = ax.imshow(drops, cmap='RdBu_r')
    for row in range(a.grid):
        for col in range(a.grid):
            ax.text(col, row, f'{drops[row, col]:.3f}', ha='center', va='center')
    fig.colorbar(plot, ax=ax, label='Decrease in foreground probability')
    fig.tight_layout()
    fig.savefig(a.out / 'occlusion.png', dpi=200)
    plt.close(fig)
if __name__ == '__main__':
    main()
