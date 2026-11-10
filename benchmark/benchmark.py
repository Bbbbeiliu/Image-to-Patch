from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
sys.dont_write_bytecode = True
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HOME = Path(__file__).resolve().parent
os.environ.setdefault('TORCH_HOME', str(HOME / 'torch_cache'))
if (HOME / 'runtime_deps').is_dir():
    sys.path.append(str(HOME / 'runtime_deps'))
ROOT = HOME.parent
PROJECT = ROOT / 'segmentation'
DEFAULT_DATA = ROOT / 'data/clinical'
PROTOCOL = {'input': 'RGB /255; 256x256 aspect-preserving centered zero letterbox', 'interpolation': 'image bilinear; binary mask nearest', 'threshold': 'sigmoid(logit)>0.5; no postprocessing', 'evaluation': 'unpadded network-resolution valid pixels; image-macro mean', 'empty_policy': 'empty GT and prediction: Dice/IoU NaN; undefined precision/recall NaN', 'loss': 'full padded canvas BCEWithLogits + batch-global soft Dice; coefficient 1'}

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def runtime():
    names = ['torch', 'torchvision', 'segmentation-models-pytorch', 'albumentations', 'numpy', 'scipy', 'opencv-python', 'matplotlib', 'timm', 'ml-collections', 'absl-py']
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': sys.version, 'executable': sys.executable, 'packages': versions}

def snapshot(data):
    data = Path(data).resolve()
    manifest = json.loads((data / '_audit/manifest.json').read_text(encoding='utf-8-sig'))
    mapping = {}
    for row in manifest:
        sid = str(row['id'])
        if sid in mapping:
            raise ValueError('Duplicate manifest sample ID')
        group = hashlib.sha256(str(row['group']).encode()).hexdigest()[:20]
        mapping[sid] = (row['split'], group)
    records = []
    group_splits = {}
    image_hash_splits = {}
    for split in ('train', 'val', 'test'):
        expected = []
        files = sorted((p for p in (data / split / 'images').iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')))
        for image in files:
            sid = image.stem
            if sid not in mapping or mapping[sid][0] != split:
                raise ValueError(f'Manifest/split mismatch: {sid}')
            group = mapping[sid][1]
            mask = data / split / 'masks' / (sid + '.json')
            if not mask.is_file():
                raise FileNotFoundError(f'Missing locked Labelme annotation: {sid}')
            h = sha(image)
            records.append({'sample_id': sid, 'split': split, 'group': group, 'image': image.relative_to(data).as_posix(), 'mask': mask.relative_to(data).as_posix(), 'image_sha256': h, 'mask_sha256': sha(mask)})
            expected.append(image.relative_to(data).as_posix())
            group_splits.setdefault(group, set()).add(split)
            image_hash_splits.setdefault(h, set()).add(split)
        listed = (data / '_audit' / (split + '.txt')).read_text(encoding='utf-8-sig').splitlines()
        listed = [s.strip().replace('\\', '/') for s in listed if s.strip()]
        if len(listed) != len(set(listed)) or set(listed) != set(expected):
            raise ValueError(f'Fixed {split}.txt does not match actual image set')
    if len(records) != len(mapping) or len({r['sample_id'] for r in records}) != len(records):
        raise ValueError('Manifest IDs differ from actual files')
    if any((len(x) != 1 for x in group_splits.values())):
        raise ValueError('Group crosses splits')
    if any((len(x) != 1 for x in image_hash_splits.values())):
        raise ValueError('Identical image crosses splits')
    payload = json.dumps(records, sort_keys=True, separators=(',', ':')).encode()
    code = {}
    for rel in ['data/custom_dataset_sam.py', 'letterbox.py', 'models/loss.py']:
        code[rel] = sha(PROJECT / rel)
    return {'dataset_fingerprint': hashlib.sha256(payload).hexdigest(), 'counts': {s: sum((r['split'] == s for r in records)) for s in ('train', 'val', 'test')}, 'groups': {s: len({r['group'] for r in records if r['split'] == s}) for s in ('train', 'val', 'test')}, 'records': records, 'protocol': PROTOCOL, 'shared_code_sha256': code, 'limitation': 'Groups are recorded separation proxies, not verified independent patient IDs.'}

def check_lock(data, lock):
    saved = json.loads(Path(lock).read_text(encoding='utf-8'))
    current = snapshot(data)
    for key in ('dataset_fingerprint', 'shared_code_sha256', 'protocol', 'records', 'counts', 'groups'):
        if current[key] != saved[key]:
            raise ValueError(f'Frozen dataset/protocol changed: {key}; do not silently regenerate lock')
    return saved

def new_output(path):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f'Output already exists; choose a new run directory: {path}')
    path.mkdir(parents=True)
    return path

def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def project_modules():
    from models import build_model, model_spec
    sys.path.append(str(PROJECT))
    from data.custom_dataset_sam import CustomDatasetWithSAM, train_transform, val_test_transform, _imread
    from letterbox import unpad
    return (build_model, model_spec, CustomDatasetWithSAM, train_transform, val_test_transform, _imread, unpad)

def datasets(data):
    _, _, cls, train_t, eval_t, _, _ = project_modules()
    result = {}
    for split in ('train', 'val', 'test'):
        b = Path(data) / split
        result[split] = cls(str(b / 'images'), str(b / 'masks'), transform=train_t if split == 'train' else eval_t)
    return result

def loss_fn(logits, targets):
    import torch
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    p = logits.sigmoid()
    return bce + 1 - (2 * (p * targets).sum() + 1e-06) / (p.sum() + targets.sum() + 1e-06)

def model_kwargs(args, training):
    return dict(encoder_weights=args.encoder_weights if training else None, transunet_root=args.transunet_root, transunet_pretrained=args.transunet_pretrained if training else None)

class CompleteBatchSampler:

    def __init__(self, size, batch_size, generator):
        if size < 2 or batch_size < 2:
            raise ValueError('Training requires >=2 cases and batch_size>=2 for BatchNorm')
        self.size, self.batch_size, self.generator = (size, batch_size, generator)

    def __iter__(self):
        import torch
        order = torch.randperm(self.size, generator=self.generator).tolist()
        batches = [order[i:i + self.batch_size] for i in range(0, self.size, self.batch_size)]
        if len(batches[-1]) == 1:
            if len(batches[-2]) >= 3:
                batches[-1].insert(0, batches[-2].pop())
            else:
                batches[-2].extend(batches.pop())
        yield from batches

    def __len__(self):
        import math
        n = math.ceil(self.size / self.batch_size)
        return n - int(self.batch_size == 2 and self.size % 2 == 1 and (n > 1))

def _legacy_fixed_budget_train(args):
    import torch
    from torch.utils.data import DataLoader
    if args.model == 'retained':
        raise ValueError('retained is evaluation-only; original prototype training is unchanged')
    if min(args.epochs, args.batch_size, args.lr) <= 0:
        raise ValueError('epochs, batch size and lr must be positive')
    frozen = check_lock(args.data, args.lock)
    seed_all(args.seed)
    build, spec, *_ = project_modules()
    ds = datasets(args.data)
    ds['train'].transform.augment.set_random_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {'train': DataLoader(ds['train'], batch_sampler=CompleteBatchSampler(len(ds['train']), args.batch_size, generator), num_workers=0), 'val': DataLoader(ds['val'], batch_size=args.batch_size, shuffle=False, num_workers=0)}
    model = build(args.model, **model_kwargs(args, True)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    out = new_output(args.out)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(dataset_fingerprint=frozen['dataset_fingerprint'], protocol=PROTOCOL, runtime=runtime(), model_spec=spec(args.model, **model_kwargs(args, True)), runner_sha256=sha(__file__), factory_sha256=sha(HOME / 'models.py'), training_policy='fixed epoch budget; checkpoint selected only by minimum validation loss', training_batch_policy='all cases once per epoch; final singleton rebalanced to non-singleton batches', external_clinical_support=False)
    if args.encoder_weights:
        init_file = Path(torch.hub.get_dir()) / 'checkpoints/resnet34-b627a593.pth'
        if not init_file.is_file():
            raise FileNotFoundError('Expected ImageNet ResNet34 initialization cache is missing')
        config['encoder_pretrained_sha256'] = sha(init_file)
    if args.transunet_pretrained:
        config['pretrained_file_sha256'] = sha(args.transunet_pretrained)
    save_json(out / 'config.json', config)
    save_json(out / 'dataset_lock.json', frozen)
    best = float('inf')
    with (out / 'history.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_loss', 'val_loss', 'seconds', 'best'])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            losses = {}
            for split in ('train', 'val'):
                model.train(split == 'train')
                total = 0.0
                for x, y, _ in loaders[split]:
                    x, y = (x.to(args.device), y.to(args.device))
                    with torch.set_grad_enabled(split == 'train'):
                        logits = model(x)
                        if logits.shape != y.shape:
                            raise ValueError(f'Output/target mismatch: {logits.shape}, {y.shape}')
                        loss = loss_fn(logits, y)
                        if not torch.isfinite(loss):
                            raise ValueError('Nonfinite loss')
                        if split == 'train':
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            optimizer.step()
                    total += loss.item() * len(x)
                losses[split] = total / len(ds[split])
            improved = losses['val'] < best
            if improved:
                best = losses['val']
            state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'best_val_loss': best, 'model': args.model, 'seed': args.seed, 'dataset_fingerprint': frozen['dataset_fingerprint'], 'protocol': PROTOCOL, 'config': config}
            for name in ['last.pt', 'best.pt'] if improved else ['last.pt']:
                temp = out / (name + '.tmp')
                torch.save(state, temp)
                os.replace(temp, out / name)
            row = dict(epoch=epoch, train_loss=losses['train'], val_loss=losses['val'], seconds=time.perf_counter() - start, best=improved)
            writer.writerow(row)
            stream.flush()
            print(json.dumps(row), flush=True)
    check_lock(args.data, args.lock)
    save_json(out / 'completed.json', {'epochs': args.epochs, 'best_val_loss': best, 'inputs_unchanged': True})

def train(args):
    sys.modules.setdefault('benchmark', sys.modules[__name__])
    from training import train as train_with_recovery
    return train_with_recovery(args)

def case_metrics(pred, gt):
    import numpy as np
    tp, fp = (int((pred & gt).sum()), int((pred & ~gt).sum()))
    fn, tn = (int((~pred & gt).sum()), int((~pred & ~gt).sum()))
    ratio = lambda a, b: a / b if b else float('nan')
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, Dice=ratio(2 * tp, 2 * tp + fp + fn), IoU=ratio(tp, tp + fp + fn), Precision=ratio(tp, tp + fp), Recall=ratio(tp, tp + fn), gt_pixels=int(gt.sum()), valid_pixels=int(gt.size), empty_prediction=not bool(np.any(pred)), empty_reference=not bool(np.any(gt)))

def evaluate(args):
    import numpy as np
    import torch
    import cv2
    from diagnostics import case_diagnostics, probability_histograms, summarize_histograms
    from evidence import benchmark_forward, write_inventory, export_plots_and_failures, check_case_exports
    torch.set_num_threads(int(getattr(args, 'torch_threads', 4)))
    frozen = check_lock(args.data, args.lock)
    seed_all(args.seed)
    build, spec, _, _, _, imread, unpad = project_modules()
    checkpoint_hash = sha(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if args.model == 'retained':
        state = checkpoint
        checkpoint_seed = None
    else:
        for key, value in (('model', args.model), ('dataset_fingerprint', frozen['dataset_fingerprint']), ('protocol', PROTOCOL)):
            if checkpoint.get(key) != value:
                raise ValueError(f'Checkpoint provenance mismatch: {key}')
        state = checkpoint['state_dict']
        checkpoint_seed = checkpoint['seed']
    model = build(args.model, **model_kwargs(args, False))
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval()
    ds = datasets(args.data)[args.split]
    mapping = {r['sample_id']: r for r in frozen['records'] if r['split'] == args.split}
    if {Path(x).stem for x in ds.image_names} != set(mapping):
        raise ValueError('Evaluation case IDs differ from lock')
    out = new_output(args.out)
    (out / 'predictions').mkdir()
    for folder in ('ground_truth', 'case_images', 'logits'):
        (out / folder).mkdir()
    if args.probabilities:
        (out / 'probabilities').mkdir()
    rows = []
    geometry_rows = []
    histogram_fg, histogram_bg = ([], [])
    timings = []
    with torch.inference_mode():
        for i, name in enumerate(ds.image_names):
            sid = Path(name).stem
            x, y, _ = ds[i]
            if str(args.device).startswith('cuda'):
                torch.cuda.synchronize()
            started = time.perf_counter()
            logits = model(x.unsqueeze(0).to(args.device))
            if str(args.device).startswith('cuda'):
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
            if tuple(logits.shape) != (1, 1, 256, 256):
                raise ValueError('Unexpected model output shape')
            prob = logits.sigmoid()[0, 0].cpu().numpy()
            hw = imread(str(Path(args.data) / mapping[sid]['image'])).shape[:2]
            prob = unpad(prob, hw)
            raw_logits = unpad(logits[0, 0].cpu().numpy(), hw).astype(np.float32)
            gt = unpad(y[0].numpy(), hw) > 0.5
            if not np.isfinite(prob).all() or not np.isfinite(raw_logits).all():
                raise ValueError('Nonfinite prediction')
            pred = prob > 0.5
            row = {'sample_id': sid, 'group': mapping[sid]['group'], **case_metrics(pred, gt)}
            row.update(case_diagnostics(pred, gt, prob))
            rows.append(row)
            ok, encoded = cv2.imencode('.png', pred.astype(np.uint8) * 255)
            if not ok:
                raise RuntimeError('PNG encoding failed')
            encoded.tofile(str(out / 'predictions' / (sid + '.png')))
            for folder, array in (('ground_truth', gt.astype(np.uint8) * 255), ('case_images', cv2.cvtColor(unpad((x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8), hw), cv2.COLOR_RGB2BGR))):
                ok, encoded = cv2.imencode('.png', array)
                if not ok:
                    raise RuntimeError('Evidence PNG encoding failed')
                encoded.tofile(str(out / folder / (sid + '.png')))
            np.save(out / 'logits' / (sid + '.npy'), raw_logits)
            from letterbox import geometry
            top, left, nh, nw = geometry(hw)
            geometry_rows.append({'sample_id': sid, 'group': mapping[sid]['group'], 'decoded_original_height': int(hw[0]), 'decoded_original_width': int(hw[1]), 'network_height': 256, 'network_width': 256, 'padding_top': top, 'padding_left': left, 'valid_height': nh, 'valid_width': nw, 'original_pixels_per_network_y': hw[0] / nh, 'original_pixels_per_network_x': hw[1] / nw, 'image_relative_to_data': mapping[sid]['image'], 'mask_relative_to_data': mapping[sid]['mask'], 'image_sha256': mapping[sid]['image_sha256'], 'annotation_sha256': mapping[sid]['mask_sha256'], 'foreground_fraction': float(gt.mean()), 'physical_calibration': None})
            hfg, hbg = probability_histograms(prob, gt, bins=4096)
            histogram_fg.append(hfg)
            histogram_bg.append(hbg)
            if args.probabilities:
                np.save(out / 'probabilities' / (sid + '.npy'), prob.astype(np.float32))
            if (i + 1) % 25 == 0 or i + 1 == len(ds):
                print(f'Evaluation {args.model}: {i + 1}/{len(ds)} cases exported', flush=True)
    with (out / 'per_case_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    if args.match_csv:
        with Path(args.match_csv).open(encoding='utf-8-sig') as stream:
            old = {r['sample_id']: r for r in csv.DictReader(stream)}
        if set(old) != {r['sample_id'] for r in rows}:
            raise ValueError('Retained comparison ID mismatch')
        for row in rows:
            if any((int(old[row['sample_id']][k]) != row[k] for k in ('tp', 'fp', 'fn', 'tn'))):
                raise ValueError(f"Retained confusion counts changed: {row['sample_id']}")
    summary = {}
    for metric in ('Dice', 'IoU', 'Precision', 'Recall', 'background_Dice', 'background_IoU', 'HD95_network_px', 'ASSD_network_px', 'BoundaryF1_1px'):
        if metric not in rows[0]:
            continue
        vals = np.asarray([r[metric] for r in rows])
        finite = vals[np.isfinite(vals)]
        summary[metric] = {'mean': float(finite.mean()) if len(finite) else None, 'sample_sd': float(finite.std(ddof=1)) if len(finite) > 1 else None, 'defined_n': int(len(finite))}
    check_lock(args.data, args.lock)
    if sha(args.checkpoint) != checkpoint_hash:
        raise RuntimeError('Checkpoint changed during evaluation')
    save_json(out / 'case_geometry.json', geometry_rows)
    np.savez_compressed(out / 'probability_histograms.npz', sample_ids=np.asarray([r['sample_id'] for r in rows]), foreground=np.asarray(histogram_fg), background=np.asarray(histogram_bg), bin_edges=np.linspace(0.0, 1.0, 4097))
    save_json(out / 'pooled_roc_pr.json', summarize_histograms(np.sum(histogram_fg, axis=0), np.sum(histogram_bg, axis=0)))
    save_json(out / 'forward_benchmark.json', benchmark_forward(model, args.device, seed=args.seed))
    with (out / 'case_forward_times.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', 'forward_seconds', 'note'])
        writer.writerows(((r['sample_id'], t, 'synchronized model forward only; actual un-warmed sequence') for r, t in zip(rows, timings)))
    frozen_code = out / 'source_snapshot'
    frozen_code.mkdir()
    import shutil
    for name in ('benchmark.py', 'models.py', 'training.py', 'diagnostics.py', 'evidence.py', 'compare.py', 'analyze_saved.py'):
        shutil.copyfile(HOME / name, frozen_code / name)
    for rel in ('data/custom_dataset_sam.py', 'letterbox.py', 'models/resnet34_unet_attention.py', 'models/fss_module.py', 'models/loss.py'):
        target = frozen_code / 'primary_project' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / rel, target)
    shutil.copyfile(HOME / 'vanilla_unet.py', frozen_code / 'vanilla_unet.py')
    shutil.copyfile(args.lock, out / 'dataset_lock.json')
    save_json(out / 'evaluation.json', {'model': args.model, 'model_spec': spec(args.model, **model_kwargs(args, False)), 'seed': checkpoint_seed, 'split': args.split, 'n': len(rows), 'dataset_fingerprint': frozen['dataset_fingerprint'], 'protocol': PROTOCOL, 'checkpoint_sha256': checkpoint_hash, 'summary': summary, 'runtime': runtime(), 'checkpoint_location': str(Path(args.checkpoint).resolve()), 'runner_sha256': sha(__file__), 'factory_sha256': sha(HOME / 'models.py'), 'diagnostics_sha256': sha(HOME / 'diagnostics.py'), 'evidence_sha256': sha(HOME / 'evidence.py'), 'evidence_version': 2, 'probabilities_exported': bool(args.probabilities), 'boundary_metric_units': 'un-padded network pixels; not physical millimeters', 'inputs_unchanged': True, 'retained_counts_matched': bool(args.match_csv), 'retained_checkpoint_provenance': 'plain state dict; see independent audit' if args.model == 'retained' else None})
    export_plots_and_failures(out)
    check_case_exports(out)
    write_inventory(out)
    print(json.dumps(summary, indent=2), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    lock = sub.add_parser('lock', help='Freeze current images, annotations, group mapping and loader')
    lock.add_argument('--data', type=Path, default=DEFAULT_DATA)
    lock.add_argument('--out', type=Path, required=True)
    env = sub.add_parser('environment', help='Read-only dependency and GPU check')
    smoke = sub.add_parser('smoke', help='Synthetic forward/backward only; no data training')
    for command in ('train', 'evaluate', 'smoke'):
        p = smoke if command == 'smoke' else sub.add_parser(command)
        p.add_argument('--model', choices=['unet', 'unet_resnet34', 'unetpp', 'deeplabv3plus', 'transunet', 'retained'], required=True)
        p.add_argument('--device', default='cuda')
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--encoder-weights', choices=['imagenet'], default=None)
        p.add_argument('--transunet-root', type=Path)
        p.add_argument('--transunet-pretrained', type=Path)
        if command != 'smoke':
            p.add_argument('--data', type=Path, default=DEFAULT_DATA)
            p.add_argument('--lock', type=Path, default=HOME / 'dataset_lock.json')
            p.add_argument('--out', type=Path, required=True)
        if command == 'train':
            p.add_argument('--epochs', type=int, default=100)
            p.add_argument('--batch-size', type=int, default=4)
            p.add_argument('--lr', type=float, default=0.0001)
            p.add_argument('--patience', type=int, default=5)
            p.add_argument('--min-delta', type=float, default=0.0)
            p.add_argument('--checkpoint-seconds', type=float, default=120.0)
            p.add_argument('--torch-threads', type=int, default=4)
            p.add_argument('--resume', action='store_true')
            p.add_argument('--stop-file', type=Path)
            p.add_argument('--show-plot', action='store_true')
        if command == 'evaluate':
            p.add_argument('--checkpoint', type=Path, required=True)
            p.add_argument('--split', choices=['val', 'test'], default='test')
            p.add_argument('--probabilities', action='store_true')
            p.add_argument('--match-csv', type=Path)
    args = parser.parse_args()
    if args.command == 'lock':
        if args.out.exists():
            raise FileExistsError('Refusing to overwrite a dataset lock')
        result = snapshot(args.data)
        save_json(args.out, result)
        print(json.dumps({k: result[k] for k in ('dataset_fingerprint', 'counts', 'groups')}, indent=2))
    elif args.command == 'environment':
        import torch
        print(json.dumps({**runtime(), 'cuda': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, indent=2))
    elif args.command == 'smoke':
        import torch
        seed_all(args.seed)
        build, *_ = project_modules()
        model = build(args.model, **model_kwargs(args, False)).to(args.device).train()
        x = torch.rand(2, 3, 256, 256, device=args.device)
        y = (torch.rand(2, 1, 256, 256, device=args.device) > 0.95).float()
        logits = model(x)
        assert logits.shape == y.shape
        loss = loss_fn(logits, y)
        assert torch.isfinite(loss)
        loss.backward()
        assert all((torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        print(json.dumps({'model': args.model, 'shape': list(logits.shape), 'loss': loss.item(), 'finite_gradients': True, 'real_training': False}))
    elif args.command == 'train':
        train(args)
    else:
        evaluate(args)
if __name__ == '__main__':
    main()
