from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

def cpu_name():
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 'HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0') as key:
            return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    except (ImportError, OSError):
        return platform.processor()

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def validate_inventory(directory):
    directory = Path(directory).resolve()
    inventory = json.loads((directory / 'artifact_inventory.json').read_text(encoding='utf-8'))
    for item in inventory['files']:
        path = (directory / item['path']).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError('Evidence inventory path is missing or outside its directory')
        if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError(f"Evidence artifact changed: {item['path']}")
    return inventory

def benchmark_forward(model, device, seed=42, warmup=20, repeats=100):
    import numpy as np
    import torch
    cuda = str(device).startswith('cuda')
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.rand(1, 3, 256, 256, device=device, generator=generator)
    model.eval()
    used = set()
    hooks = []

    def track(module, _inputs, _outputs):
        used.update((id(p) for p in module.parameters(recurse=False)))
    for module in model.modules():
        hooks.append(module.register_forward_hook(track))
    try:
        with torch.inference_mode():
            model(x)
    finally:
        for hook in hooks:
            hook.remove()
    if cuda:
        torch.cuda.synchronize()
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()
        else:
            baseline = None
        durations = []
        for _ in range(repeats):
            if cuda:
                torch.cuda.synchronize()
            started = time.perf_counter()
            model(x)
            if cuda:
                torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
    return {'protocol': 'eval/inference_mode; float32; batch1x3x256x256; no autocast; synchronized forward only', 'excludes': ['image decoding', 'preprocessing', 'CPU-to-GPU transfer', 'postprocessing', 'network/application/operator time'], 'warmup': warmup, 'repeats': repeats, 'seed': seed, 'seconds': durations, 'mean_ms': float(np.mean(durations) * 1000), 'median_ms': float(np.median(durations) * 1000), 'p95_ms': float(np.quantile(durations, 0.95) * 1000), 'images_per_second_from_mean': float(1 / np.mean(durations)), 'serialized_parameters': sum((p.numel() for p in model.parameters())), 'executed_module_parameters': sum((p.numel() for p in model.parameters() if id(p) in used)), 'parameter_count_method': 'Unique parameters in executed modules, tracked by forward hooks; bypassed modules excluded from executed count', 'cuda_allocated_before_timing_bytes': baseline, 'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if cuda else None, 'cuda_peak_increment_bytes': torch.cuda.max_memory_allocated() - baseline if cuda else None, 'gpu': torch.cuda.get_device_name(device) if cuda else None, 'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'tf32_matmul_allowed': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn_allowed': torch.backends.cudnn.allow_tf32, 'torch_threads': torch.get_num_threads(), 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(), 'os': platform.platform(), 'cpu': cpu_name(), 'limitation': 'Inference-only measurements on the rerun device; not full clinical workflow timing or a FLOP estimate.'}

def _load_image(path):
    import cv2
    import numpy as np
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f'Unreadable evidence image: {Path(path).name}')
    return image

def export_plots_and_failures(directory):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    curves = json.loads((directory / 'pooled_roc_pr.json').read_text(encoding='utf-8'))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(curves['fpr'], curves['tpr'])
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='Binned pooled-pixel ROC')
    axes[1].plot(curves['recall'], curves['precision'])
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Binned pooled-pixel precision–recall')
    fig.suptitle('4096 bins; dependent pixels; raw float32 scores retained')
    fig.tight_layout()
    fig.savefig(directory / 'pooled_roc_pr.png', dpi=160)
    plt.close(fig)
    with (directory / 'per_case_metrics.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    ordered = sorted(rows, key=lambda r: (float(r['Dice']), r['sample_id']))
    choices = [('lowest', r) for r in ordered[:3]]
    choices.append(('median', ordered[len(ordered) // 2]))
    fig, axes = plt.subplots(len(choices), 4, figsize=(12, 3 * len(choices)), squeeze=False)
    selected = []
    for i, (reason, row) in enumerate(choices):
        sid = row['sample_id']
        image = _load_image(directory / 'case_images' / (sid + '.png'))[:, :, ::-1]
        gt = _load_image(directory / 'ground_truth' / (sid + '.png')) > 0
        pred = _load_image(directory / 'predictions' / (sid + '.png')) > 0
        errors = np.zeros((*gt.shape, 3), dtype=np.uint8)
        errors[pred & gt] = [0, 180, 0]
        errors[pred & ~gt] = [230, 30, 30]
        errors[~pred & gt] = [40, 100, 230]
        for axis, array, title in zip(axes[i], (image, gt, pred, errors), (f'{sid} ({reason})', 'Reference', f"Prediction Dice={float(row['Dice']):.3f}", 'TP green / FP red / FN blue')):
            axis.imshow(array, cmap='gray' if array.ndim == 2 else None, vmin=0, vmax=1 if array.ndim == 2 else 255)
            axis.set_title(title, fontsize=9)
            axis.axis('off')
        selected.append({'sample_id': sid, 'selection': reason, 'Dice': float(row['Dice'])})
    fig.suptitle('Objectively selected cases; unpadded network resolution; not clinical acceptability thresholds')
    fig.tight_layout()
    fig.savefig(directory / 'failure_cases.png', dpi=160)
    plt.close(fig)
    (directory / 'failure_selection.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')

def write_inventory(directory):
    directory = Path(directory)
    records = []
    for path in sorted(directory.rglob('*')):
        if path.is_file() and path.name != 'artifact_inventory.json':
            records.append({'path': path.relative_to(directory).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest(path)})
    result = {'schema_version': 2, 'files': records, 'data_preservation': 'Keep original locked dataset as well as this evidence directory; case_images are network-resolution copies, not original photographs.', 'supports_without_training': ['paired Dice/IoU and class-wise statistics', 'cluster bootstrap and signed-rank tests', 'case scatter and difference plots', 'boundary-error summaries in network pixels', 'ROC/PR and threshold/reliability analysis from saved probabilities/logits', 'failure and subgroup analyses', 'inference-efficiency comparison'], 'may_require_new_inference_but_not_training': ['Grad-CAM', 'occlusion/LIME/SHAP for new perturbations', 'alternative input-resolution inference'], 'not_created_by_archiving': ['new independent training seeds', 'cross-validation folds', 'new architectures or component ablations', 'missing patient identities/demographics', 'physical calibration or clinical outcome evidence']}
    (directory / 'artifact_inventory.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result

def check_case_exports(directory):
    import numpy as np
    directory = Path(directory)
    meta = json.loads((directory / 'evaluation.json').read_text(encoding='utf-8'))
    geometry = json.loads((directory / 'case_geometry.json').read_text(encoding='utf-8'))
    geo = {r['sample_id']: r for r in geometry}
    with (directory / 'per_case_metrics.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    ids = {r['sample_id'] for r in rows}
    if len(rows) != len(ids) or len(ids) != meta['n'] or set(geo) != ids:
        raise ValueError('Evidence case IDs/counts do not match')
    folders = {'predictions': '.png', 'ground_truth': '.png', 'case_images': '.png', 'logits': '.npy'}
    if meta['probabilities_exported']:
        folders['probabilities'] = '.npy'
    for name, ext in folders.items():
        if {p.stem for p in (directory / name).glob('*' + ext)} != ids:
            raise ValueError(f'Incomplete case export: {name}')
    for row in rows:
        sid = row['sample_id']
        pred = _load_image(directory / 'predictions' / (sid + '.png')) > 0
        gt = _load_image(directory / 'ground_truth' / (sid + '.png')) > 0
        logits = np.load(directory / 'logits' / (sid + '.npy'), allow_pickle=False)
        expected = (geo[sid]['valid_height'], geo[sid]['valid_width'])
        if pred.shape != expected or gt.shape != expected or logits.shape != expected or (not np.isfinite(logits).all()):
            raise ValueError(f'Evidence geometry/finite-score mismatch: {sid}')
        counts = dict(tp=int((pred & gt).sum()), fp=int((pred & ~gt).sum()), fn=int((~pred & gt).sum()), tn=int((~pred & ~gt).sum()))
        if any((int(row[k]) != v for k, v in counts.items())):
            raise ValueError(f'Stored mask/count mismatch: {sid}')
        if meta['probabilities_exported']:
            prob = np.load(directory / 'probabilities' / (sid + '.npy'), allow_pickle=False)
            if prob.shape != expected or not np.isfinite(prob).all() or (not np.array_equal(prob > 0.5, pred)):
                raise ValueError(f'Stored probability/prediction mismatch: {sid}')
    result = {'status': 'passed', 'n': len(rows), 'case_ids_aligned': True, 'binary_masks_match_confusion_counts': True, 'full_precision_scores_and_geometry_verified': True, 'folder_counts': {k: len(ids) for k in folders}}
    (directory / 'completeness.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result

def finalize_suite(session, models, reference, comparison):
    session, reference, comparison = (Path(session), Path(reference), Path(comparison))
    checks = []
    for name in [*models, 'reference']:
        base = session / name
        if name != 'reference':
            for filename in ('best.pt', 'last.pt', 'history.csv', 'completed.json', 'training_curves.png', 'config.json', 'progress_history.jsonl'):
                if not (base / filename).is_file():
                    raise ValueError(f'Missing training record: {name}/{filename}')
        for split in ('val', 'test'):
            validate_inventory(base / split)
            checked = json.loads((base / split / 'completeness.json').read_text(encoding='utf-8'))
            if checked['status'] != 'passed':
                raise ValueError('Case evidence verification did not pass')
            if not (base / (split + '_analysis') / 'summary.json').is_file():
                raise ValueError(f'Missing offline metric analysis: {name}/{split}')
            checks.append({'model': name, 'split': split, 'n': checked['n'], 'passed': True})
    for filename in ('summary.json', 'summary.csv'):
        if not (comparison / filename).is_file():
            raise ValueError('Missing paired comparison summary')
    for model in models:
        for filename in ('score_scatter.png', 'bland_altman_style.png', 'cases_ranked.csv'):
            if not (comparison / ('pair_' + model) / filename).is_file():
                raise ValueError(f'Missing paired comparison: {model}/{filename}')
    result = {'status': 'analysis_ready', 'evaluations': checks, 'weights_and_training_history_present': True, 'paired_statistics_and_figures_present': True, 'no_retraining_for_saved_model_posthoc_analysis': True, 'retain_original_locked_dataset': True, 'limits': ['New folds/seeds/architectures/ablations require new experiments.', 'Physical millimeter validation needs true calibration; missing clinical evidence cannot be inferred from masks.', 'New explanation perturbations may require inference/gradients with saved weights, but no optimization.']}
    (session / 'analysis_ready.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
