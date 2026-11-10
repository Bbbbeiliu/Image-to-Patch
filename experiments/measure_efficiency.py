import argparse
import gc
import json
import os
from pathlib import Path
import random
import sys

def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', type=Path, required=True, help='JSON mapping model names to local checkpoint files')
    p.add_argument('--data', type=Path, default=root / 'data/clinical')
    p.add_argument('--lock', type=Path, default=root / 'benchmark/dataset_lock.json')
    p.add_argument('--device', default='cuda')
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Choose a new output directory')
    home = root / 'benchmark'
    sys.path.insert(0, str(home))
    os.environ['SCAR_DATASET_LOCK'] = str(a.lock.resolve())
    import benchmark as b
    import torch
    import numpy as np
    from models import build_model
    from evidence import benchmark_forward
    b.DEFAULT_DATA = a.data.resolve()
    torch.set_num_threads(4)
    files = json.loads(a.checkpoints.read_text(encoding='utf-8'))
    a.out.mkdir(parents=True)
    records = []

    class FixedSupport:

        def __init__(self, base, index):
            self.base, self.index = (base, index)
            self.value = tuple((v.to(a.device) for v in base[index]))

        def __getattr__(self, key):
            return getattr(self.base, key)

        def __getitem__(self, index):
            if index != self.index:
                raise ValueError('Unexpected evaluation support')
            return self.value
    for repeat in range(a.rounds):
        order = list(files)
        random.Random(20261106 + repeat).shuffle(order)
        for name in order:
            b.seed_all(42)
            if name == 'manet':
                from extended_models import build_model as build
                model = build('manet', encoder_weights=None)
            elif name == 'panet':
                from extended_baselines.panet import adapter
                model = adapter.build_model()
            elif name == 'samed':
                from extended_baselines.samed import adapter
                model = adapter.build_model()
            else:
                model = build_model('retained' if name in ['proposed', 'reference', 'retained'] else name, encoder_weights=None, transunet_root=home / 'vendor/TransUNet' if name == 'transunet' else None)
            checkpoint = torch.load(files[name], map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint.get('state_dict', checkpoint.get('model_state', checkpoint)), strict=True)
            model.to(a.device).eval()
            if name == 'panet':
                model.support_dataset = FixedSupport(model.support_dataset, model.fixed_index)
            del checkpoint
            gc.collect()
            record = benchmark_forward(model, a.device, seed=42, warmup=a.warmup, repeats=a.repeats)
            record.update(model=name, repeat=repeat + 1)
            records.append(record)
            (a.out / f'{name}_round{repeat + 1}.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
            del model
            gc.collect()
            if a.device.startswith('cuda'):
                torch.cuda.empty_cache()
    summary = []
    for name in files:
        chosen = [r for r in records if r['model'] == name]
        ms = np.concatenate([r['seconds'] for r in chosen]) * 1000
        summary.append({'model': name, 'params_total_M': chosen[0]['serialized_parameters'] / 1000000.0, 'params_forward_M': chosen[0]['executed_module_parameters'] / 1000000.0, 'median_ms': float(np.median(ms)), 'mean_ms': float(ms.mean()), 'images_per_second': 1000 / float(ms.mean()), 'timed_calls': len(ms)})
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
if __name__ == '__main__':
    main()
