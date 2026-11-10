from __future__ import annotations
import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
from types import SimpleNamespace
from typing import Any
sys.dont_write_bytecode = True
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
CODE_DIR = Path(__file__).resolve().parent
WORKSPACE = Path(__file__).resolve().parents[1]
BENCHMARK_HOME = WORKSPACE / 'benchmark'
PROJECT = WORKSPACE / 'segmentation'
DEFAULT_DATA = WORKSPACE / 'data' / 'clinical'
DEFAULT_LOCK = BENCHMARK_HOME / 'dataset_lock.json'
DEFAULT_PUBLIC = WORKSPACE / 'data' / 'public'
ENCODER_WEIGHTS = BENCHMARK_HOME / 'torch_cache' / 'hub' / 'checkpoints' / 'resnet34-b627a593.pth'
ORIGINAL_FULL_CHECKPOINT = WORKSPACE / 'weights' / 'proposed.pth'
VARIANTS = ('backbone', 'attention_only', 'prototype_only', 'full', 'full_no_augmentation')
SEED = 42
MAX_EPOCHS = 100
PATIENCE = 5
MIN_DELTA = 0.0
BATCH_SIZE = 4
LEARNING_RATE = 0.0001
LAMBDA_PROTOTYPE = 0.5
CHECKPOINT_SECONDS = 120.0
TORCH_THREADS = 4
PUBLIC_PAIR_COUNT = 3694
MODEL_PREFIX = 'issue10_ablation_'
CHECKPOINT_FORMAT = 1
if str(BENCHMARK_HOME) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_HOME))
os.environ.setdefault('TORCH_HOME', str(BENCHMARK_HOME / 'torch_cache'))

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

def atomic_write(path: Path, writer) -> None:
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise

def atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    atomic_write(path, lambda stream: stream.write(payload))

def _runtime():
    global _SHARED, _TRAINING_HELPERS
    if '_SHARED' not in globals():
        import benchmark
        _SHARED = benchmark
        _TRAINING_HELPERS = importlib.import_module('training')
    return (_SHARED, _TRAINING_HELPERS)

def _import_primary_model_source():
    source = PROJECT / 'models' / 'resnet34_unet_attention.py'
    source_id = sha256(source)[:16]
    module_name = f'_issue10_retained_unet_{source_id}'
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not import the retained architecture: {source}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module

def _build_model(variant: str, *, initialize_imagenet: bool=True):
    import torch
    import segmentation_models_pytorch as smp
    from unittest import mock
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant: {variant}')
    if not ENCODER_WEIGHTS.is_file():
        raise FileNotFoundError(f'Expected local ResNet34 initialization is missing: {ENCODER_WEIGHTS}')
    unet_api = importlib.import_module('segmentation_models_pytorch.decoders.unet.model')
    original_get_encoder = smp.encoders.get_encoder

    def no_download_encoder(*args, **kwargs):
        kwargs['weights'] = None
        return original_get_encoder(*args, **kwargs)
    source = _import_primary_model_source()
    with mock.patch.object(smp.encoders, 'get_encoder', side_effect=no_download_encoder), mock.patch.object(unet_api, 'get_encoder', side_effect=no_download_encoder):
        model = source.UNetWithAttentionResNet34(in_channels=3, out_channels=1)
    if variant in ('backbone', 'prototype_only'):
        for name in ('attention1', 'attention2', 'attention3', 'attention4'):
            setattr(model, name, torch.nn.Identity())
    if initialize_imagenet:
        pretrained = torch.load(ENCODER_WEIGHTS, map_location='cpu', weights_only=True)
        for encoder_name, encoder in (('encoder', model.encoder), ('unet.encoder', model.unet.encoder)):
            target = encoder.state_dict()
            matched = {key: value for key, value in pretrained.items() if key in target and tuple(value.shape) == tuple(target[key].shape)}
            missing = sorted((key for key in set(target) - set(matched) if not key.endswith('num_batches_tracked')))
            unexpected = sorted((key for key in pretrained if key not in target and (not key.startswith('fc.'))))
            if missing:
                raise ValueError(f'ImageNet cache did not initialize {encoder_name}; missing={missing[:5]}')
            if unexpected:
                raise ValueError(f'Unexpected ResNet34 initialization keys for {encoder_name}: {unexpected[:5]}')
            encoder.load_state_dict(matched, strict=False)
    model.issue10_variant = variant
    model.issue10_prototype_aux = variant in ('prototype_only', 'full', 'full_no_augmentation')
    model.issue10_no_augmentation = variant == 'full_no_augmentation'
    return model

def _model_spec(variant: str) -> dict[str, Any]:
    source = PROJECT / 'models' / 'resnet34_unet_attention.py'
    return {'name': MODEL_PREFIX + variant, 'variant': variant, 'architecture': 'retained UNetWithAttentionResNet34; ResNet34 encoder; SMP U-Net decoder', 'implementation': str(source.relative_to(WORKSPACE)).replace('\\', '/'), 'source_sha256': sha256(source), 'input_shape': ['B', 3, 256, 256], 'output_shape': ['B', 1, 256, 256], 'output': 'logits', 'decoder_channels': [256, 128, 64, 32, 16], 'final_head_channels': 16, 'attention_blocks_active': variant in ('attention_only', 'full', 'full_no_augmentation'), 'prototype_auxiliary_loss': variant in ('prototype_only', 'full', 'full_no_augmentation'), 'prototype_lambda': LAMBDA_PROTOTYPE if variant in ('prototype_only', 'full', 'full_no_augmentation') else 0.0, 'prototype_encoder_output': 'model.unet.encoder(images)[-1]', 'prototype_support_count': PUBLIC_PAIR_COUNT if variant in ('prototype_only', 'full', 'full_no_augmentation') else 0, 'encoder_initialization': 'local torchvision ResNet34 ImageNet state; loaded identically into both source encoders', 'encoder_initialization_sha256': sha256(ENCODER_WEIGHTS), 'full_no_augmentation_transform': 'deterministic 256x256 letterbox only', 'original_full_checkpoint_loaded': False}

def _read_public_manifest(image_dir: Path, mask_dir: Path) -> dict[str, Any]:
    image_paths = sorted((p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')))
    records = []
    for image in image_paths:
        mask = mask_dir / f'{image.stem}_segmentation.png'
        if not mask.is_file():
            raise FileNotFoundError(f'Public support mask is missing: {mask.name}')
        records.append({'id': image.stem, 'image': image.name, 'mask': mask.name, 'image_sha256': sha256(image), 'mask_sha256': sha256(mask)})
    if len(records) != PUBLIC_PAIR_COUNT:
        raise ValueError(f'Expected all {PUBLIC_PAIR_COUNT} public image/mask pairs, found {len(records)}')
    payload = json.dumps(records, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return {'count': len(records), 'dataset_fingerprint': hashlib.sha256(payload).hexdigest(), 'image_dir': str(image_dir.resolve()), 'mask_dir': str(mask_dir.resolve()), 'records': records}

def _make_datasets(shared, data: Path, variant: str, public_data: Path | None, seed: int):
    import torch
    from torch.utils.data import DataLoader
    datasets = shared.datasets(data)
    _, _, dataset_class, train_transform, eval_transform, _, _ = shared.project_modules()
    if variant == 'full_no_augmentation':
        for split in ('train', 'val', 'test'):
            datasets[split].transform = eval_transform
        clinical_transform = eval_transform
    else:
        clinical_transform = datasets['train'].transform
    if len(datasets['train']) < 2 or len(datasets['val']) < 1 or len(datasets['test']) < 1:
        raise ValueError('The locked clinical split must contain train, validation and test examples')
    train_aug = getattr(clinical_transform, 'augment', None)
    if train_aug is not None and hasattr(train_aug, 'set_random_seed'):
        train_aug.set_random_seed(seed)
    needs_prototype = variant in ('prototype_only', 'full', 'full_no_augmentation')
    public_dataset = None
    public_loader = None
    public_generator = None
    public_manifest = None
    if needs_prototype:
        public_data = Path(public_data or DEFAULT_PUBLIC).resolve()
        public_images, public_masks = (public_data / 'images', public_data / 'masks')
        public_manifest = _read_public_manifest(public_images, public_masks)
        public_transform = eval_transform if variant == 'full_no_augmentation' else train_transform
        public_dataset = dataset_class(str(public_images), str(public_masks), sam_dir=None, transform=public_transform, mask_suffix='_segmentation.png')
        if len(public_dataset) != PUBLIC_PAIR_COUNT:
            raise ValueError(f'Public loader resolved {len(public_dataset)} items; expected {PUBLIC_PAIR_COUNT}')
        public_aug = getattr(public_transform, 'augment', None)
        if public_aug is not None and hasattr(public_aug, 'set_random_seed'):
            public_aug.set_random_seed(seed + 1)
        public_generator = torch.Generator().manual_seed(seed + 1)
        public_loader = DataLoader(public_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=public_generator, num_workers=0)
    return (datasets, public_dataset, public_loader, public_generator, public_manifest)

def _compute_global_prototype(model, loader, device):
    import torch
    import torch.nn.functional as F
    previous = model.training
    model.eval()
    feature_sum = None
    pixel_count = 0.0
    try:
        with torch.inference_mode():
            for images, masks, _ in loader:
                images = images.to(device)
                masks = masks.to(device)
                feature = model.unet.encoder(images)[-1]
                masks_ds = F.interpolate(masks, size=feature.shape[-2:], mode='bilinear', align_corners=False)
                masks_bin = (masks_ds > 0.5).to(feature.dtype)
                current = (feature * masks_bin).sum(dim=(0, 2, 3))
                feature_sum = current if feature_sum is None else feature_sum + current
                pixel_count += float(masks_bin.sum().item())
        if feature_sum is None:
            raise ValueError('The public prototype support loader is empty')
        return (feature_sum / (pixel_count + 1e-06)).view(1, -1, 1, 1).detach()
    finally:
        model.train(previous)

def _prototype_auxiliary_loss(model, images, masks, prototype, base_criterion):
    import torch.nn.functional as F
    feature = model.unet.encoder(images)[-1]
    feature_norm = F.normalize(feature, dim=1)
    prototype_norm = F.normalize(prototype, dim=1)
    similarity = (feature_norm * prototype_norm).sum(dim=1, keepdim=True)
    similarity = F.interpolate(similarity, size=masks.shape[-2:], mode='bilinear', align_corners=False)
    return base_criterion(similarity, masks)

def _rng_state(torch, np, helpers, sampler_generator, train_dataset, public_generator=None, public_dataset=None):
    state = helpers._capture_rng(torch, np, sampler_generator, train_dataset)
    state['public_sampler'] = public_generator.get_state() if public_generator is not None else None
    state['public_albumentations'] = helpers._capture_augmentation_rng(public_dataset, np) if public_dataset is not None else []
    return state

def _restore_rng(torch, np, helpers, sampler_generator, train_dataset, state, public_generator=None, public_dataset=None):
    helpers._restore_rng(torch, np, sampler_generator, train_dataset, state)
    if public_generator is not None:
        saved = state.get('public_sampler')
        if saved is None:
            raise ValueError('Checkpoint is missing the public support sampler RNG')
        public_generator.set_state(saved)
        helpers._restore_augmentation_rng(public_dataset, state.get('public_albumentations', []), np)

def _atomic_checkpoint(torch, helpers, path: Path, checkpoint: dict[str, Any]) -> None:
    helpers._atomic_torch_save(torch, path, checkpoint)

def train(args) -> Path:
    import numpy as np
    import torch
    shared, helpers = _runtime()
    if args.variant not in VARIANTS:
        raise ValueError(f"Unknown variant {args.variant!r}; choose from {', '.join(VARIANTS)}")
    out = Path(args.out).expanduser().resolve()
    try:
        out.relative_to(CODE_DIR.resolve())
    except ValueError:
        pass
    else:
        raise ValueError('Run outputs must be outside the code directory')
    if args.resume:
        if not out.is_dir() or not (out / 'last.pt').is_file():
            raise FileNotFoundError(f'No resumable run with last.pt at {out}')
        if (out / 'completed.json').exists():
            raise ValueError('This run is complete; choose a new output directory')
    elif out.exists():
        raise FileExistsError(f'Output already exists; use a new run directory: {out}')
    data = Path(args.data).expanduser().resolve()
    lock = Path(args.lock).expanduser().resolve()
    public_data = Path(args.public_data).expanduser().resolve() if args.public_data else DEFAULT_PUBLIC
    torch.set_num_threads(TORCH_THREADS)
    try:
        torch.set_num_interop_threads(TORCH_THREADS)
    except RuntimeError:
        pass
    seed = SEED
    if str(args.device).startswith('cuda') and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA was requested but is not available in this Python environment')
    frozen = shared.check_lock(data, lock)
    shared.seed_all(seed)
    ds, public_ds, public_loader, public_generator, public_manifest = _make_datasets(shared, data, args.variant, public_data, seed)
    sampler_generator = torch.Generator().manual_seed(seed)
    model_name = MODEL_PREFIX + args.variant
    spec = _model_spec(args.variant)
    source_paths = {'runner': Path(__file__), 'benchmark': BENCHMARK_HOME / 'benchmark.py', 'resume_engine': BENCHMARK_HOME / 'training.py', 'clinical_loader': PROJECT / 'data' / 'custom_dataset_sam.py', 'letterbox': PROJECT / 'letterbox.py', 'model': PROJECT / 'models' / 'resnet34_unet_attention.py', 'prototype': PROJECT / 'models' / 'fss_module.py', 'loss': PROJECT / 'models' / 'loss.py', 'imagenet_resnet34': ENCODER_WEIGHTS, 'original_full_checkpoint_evaluation_only': ORIGINAL_FULL_CHECKPOINT}
    source_hashes = {key: sha256(path) for key, path in source_paths.items() if path.is_file()}
    signature = {'variant': args.variant, 'model': model_name, 'dataset_fingerprint': frozen['dataset_fingerprint'], 'public_dataset_fingerprint': public_manifest['dataset_fingerprint'] if public_manifest else None, 'seed': seed, 'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE, 'max_epochs': MAX_EPOCHS, 'patience': PATIENCE, 'min_delta': MIN_DELTA, 'lambda_prototype': LAMBDA_PROTOTYPE if public_manifest else 0.0, 'protocol': jsonable(shared.PROTOCOL), 'model_spec': spec, 'source_hashes': source_hashes, 'device': str(args.device)}
    saved_state = None
    if args.resume:
        config = json.loads((out / 'config.json').read_text(encoding='utf-8'))
        if config.get('training_signature') != signature:
            raise ValueError('Resume signature differs in code, data, variant, initialization or training protocol')
        saved_lock = json.loads((out / 'dataset_lock.json').read_text(encoding='utf-8'))
        if saved_lock.get('dataset_fingerprint') != frozen['dataset_fingerprint']:
            raise ValueError('Saved clinical dataset lock differs from the current lock')
        if public_manifest:
            saved_public = json.loads((out / 'public_dataset_manifest.json').read_text(encoding='utf-8'))
            if saved_public.get('dataset_fingerprint') != public_manifest['dataset_fingerprint']:
                raise ValueError('Saved public support data differs from the current 3694-pair set')
        saved_state = torch.load(out / 'last.pt', map_location='cpu', weights_only=True)
        if saved_state.get('checkpoint_format') != CHECKPOINT_FORMAT or saved_state.get('training_signature') != signature:
            raise ValueError('Unsupported or mismatched resume checkpoint')
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
    model = _build_model(args.variant).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if args.resume:
        model.load_state_dict(saved_state['model_state'], strict=True)
        optimizer.load_state_dict(saved_state['optimizer_state'])
        rng = saved_state['rng_state']
        _restore_rng(torch, np, helpers, sampler_generator, ds['train'], rng, public_generator, public_ds)
        history = list(saved_state['history'])
        best_val = float(saved_state['best_val_loss'])
        best_epoch = int(saved_state['best_epoch'])
        bad_epochs = int(saved_state['bad_epochs'])
        active_epoch = saved_state['active_epoch']
        phase = saved_state['phase']
        active_batches = saved_state['active_batches']
        next_batch_index = int(saved_state['next_batch_index'])
        train_loss_sum = float(saved_state['train_loss_sum'])
        train_case_count = int(saved_state['train_case_count'])
        val_batch_index = int(saved_state['val_batch_index'])
        val_loss_sum = float(saved_state['val_loss_sum'])
        val_case_count = int(saved_state['val_case_count'])
        prototype = saved_state.get('prototype')
        prototype = prototype.to(args.device) if prototype is not None else None
        prototype_epoch = int(saved_state.get('prototype_epoch', 0))
        elapsed_base = float(saved_state.get('elapsed_seconds', 0.0))
        epoch_elapsed_base = float(saved_state.get('active_epoch_elapsed_seconds', 0.0))
        config['resume_count'] = int(config.get('resume_count', 0)) + 1
        config['last_resumed_at'] = time.time()
        atomic_json(out / 'config.json', config)
        helpers._atomic_history(out / 'history.csv', history)
    else:
        history = []
        best_val = float('inf')
        best_epoch = 0
        bad_epochs = 0
        active_epoch = 1
        phase = 'train'
        active_batches = None
        next_batch_index = 0
        train_loss_sum = 0.0
        train_case_count = 0
        val_batch_index = 0
        val_loss_sum = 0.0
        val_case_count = 0
        prototype = None
        prototype_epoch = 0
        elapsed_base = 0.0
        epoch_elapsed_base = 0.0
        out.mkdir(parents=True, exist_ok=False)
        config = {'arguments': {'variant': args.variant, 'data': str(data), 'lock': str(lock), 'out': str(out), 'public_data': str(public_data), 'device': str(args.device), 'resume': False}, 'training_signature': signature, 'dataset_fingerprint': frozen['dataset_fingerprint'], 'protocol': jsonable(shared.PROTOCOL), 'runtime': shared.runtime(), 'model_spec': spec, 'source_paths': {key: str(path.resolve()) for key, path in source_paths.items() if path.is_file()}, 'source_hashes': source_hashes, 'max_epochs': MAX_EPOCHS, 'patience': PATIENCE, 'min_delta': MIN_DELTA, 'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE, 'lambda_prototype': signature['lambda_prototype'], 'checkpoint_seconds': CHECKPOINT_SECONDS, 'torch_threads': TORCH_THREADS, 'seed': seed, 'training_policy': 'Adam; at most 100 epochs; early stopping after 5 non-improving validation epochs; min_delta=0', 'initialization_policy': 'shared ImageNet ResNet34 cache plus newly initialized decoder/head/attention; no main checkpoint initialization', 'external_support': 'all 3694 public ISIC image/mask pairs' if public_manifest else None, 'resume_count': 0, 'created_at': time.time()}
        atomic_json(out / 'config.json', config)
        atomic_json(out / 'dataset_lock.json', frozen)
        if public_manifest:
            atomic_json(out / 'public_dataset_manifest.json', public_manifest)
    last_checkpoint_at = time.perf_counter()
    epoch_started = last_checkpoint_at
    stop_requested = {'signal': None}
    pause_state = {'reason': None}
    previous_handlers = {}

    def on_signal(signum, _frame):
        stop_requested['signal'] = int(signum)
    for sig in (getattr(signal, 'SIGINT', None), getattr(signal, 'SIGTERM', None), getattr(signal, 'SIGBREAK', None)):
        if sig is None:
            continue
        try:
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, on_signal)
        except (ValueError, OSError):
            previous_handlers.pop(sig, None)

    def checkpoint(status='running', save_best=False):
        nonlocal elapsed_base, epoch_elapsed_base, epoch_started, last_checkpoint_at
        now = time.perf_counter()
        elapsed_base += now - last_checkpoint_at
        active_epoch_elapsed = epoch_elapsed_base + (now - epoch_started) if active_epoch is not None else 0.0
        packed = {'checkpoint_format': CHECKPOINT_FORMAT, 'model': model_name, 'variant': args.variant, 'seed': seed, 'dataset_fingerprint': frozen['dataset_fingerprint'], 'protocol': jsonable(shared.PROTOCOL), 'training_signature': signature, 'model_state': model.state_dict(), 'state_dict': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'rng_state': _rng_state(torch, np, helpers, sampler_generator, ds['train'], public_generator, public_ds), 'active_epoch': active_epoch, 'phase': phase, 'active_batches': active_batches, 'next_batch_index': next_batch_index, 'train_loss_sum': train_loss_sum, 'train_case_count': train_case_count, 'val_batch_index': val_batch_index, 'val_loss_sum': val_loss_sum, 'val_case_count': val_case_count, 'completed_epochs': len(history), 'history': history, 'best_val_loss': best_val, 'best_epoch': best_epoch, 'bad_epochs': bad_epochs, 'prototype': prototype.detach().cpu() if prototype is not None else None, 'prototype_epoch': prototype_epoch, 'elapsed_seconds': elapsed_base, 'active_epoch_elapsed_seconds': active_epoch_elapsed, 'status': status, 'max_epochs': MAX_EPOCHS}
        if save_best:
            _atomic_checkpoint(torch, helpers, out / 'best.pt', packed)
        _atomic_checkpoint(torch, helpers, out / 'last.pt', packed)
        atomic_json(out / 'progress.json', {'status': status, 'phase': phase, 'epoch': active_epoch, 'max_epochs': MAX_EPOCHS, 'next_batch_index': next_batch_index, 'batches_in_epoch': len(active_batches or []), 'best_epoch': best_epoch, 'best_val_loss': best_val if best_val != float('inf') else None, 'bad_epochs': bad_epochs, 'patience': PATIENCE, 'elapsed_seconds': elapsed_base, 'last_checkpoint': str(out / 'last.pt')})
        last_checkpoint_at = time.perf_counter()
        epoch_elapsed_base = active_epoch_elapsed
        epoch_started = last_checkpoint_at

    def should_pause() -> str | None:
        if stop_requested['signal'] is not None:
            return f"signal:{stop_requested['signal']}"
        if (out / 'STOP_REQUESTED').exists():
            return 'stop_file'
        return None

    def pause(reason: str):
        pause_state['reason'] = reason
        checkpoint(status='paused')
        atomic_json(out / 'interrupted.json', {'status': 'paused', 'reason': reason, 'active_epoch': active_epoch, 'completed_epochs': len(history), 'last_checkpoint': 'last.pt', 'last_checkpoint_sha256': sha256(out / 'last.pt'), 'saved_at': time.time()})
        stop_marker = out / 'STOP_REQUESTED'
        if stop_marker.exists():
            stop_marker.unlink()
        raise KeyboardInterrupt(f'Training paused safely ({reason})')
    try:
        if not args.resume:
            checkpoint(status='running')
            helpers._atomic_history(out / 'history.csv', history)
        reason = should_pause()
        if reason:
            pause(reason)
        if prototype is None and public_loader is not None:
            phase = 'prototype'
            print('Computing initial prototype from all public support pairs', flush=True)
            prototype = _compute_global_prototype(model, public_loader, args.device)
            prototype_epoch = int(active_epoch or 1)
            phase = 'train'
            checkpoint(status='running')
        while active_epoch is not None:
            if prototype is not None and active_epoch > 1 and ((active_epoch - 1) % 5 == 0) and (prototype_epoch != active_epoch):
                phase = 'prototype'
                print(f'Refreshing public prototype before epoch {active_epoch}', flush=True)
                prototype = _compute_global_prototype(model, public_loader, args.device)
                prototype_epoch = active_epoch
                phase = 'train'
                checkpoint(status='running')
            if phase == 'train':
                if active_batches is None:
                    active_batches = helpers._make_batches(shared, len(ds['train']), BATCH_SIZE, sampler_generator)
                    epoch_elapsed_base = 0.0
                    epoch_started = time.perf_counter()
                    next_batch_index = 0
                    train_loss_sum = 0.0
                    train_case_count = 0
                    val_batch_index = 0
                    val_loss_sum = 0.0
                    val_case_count = 0
                    checkpoint(status='running')
                if not 0 <= next_batch_index <= len(active_batches):
                    raise ValueError('Saved training batch index is outside the epoch order')
                model.train()
                for batch_index in range(next_batch_index, len(active_batches)):
                    reason = should_pause()
                    if reason:
                        pause(reason)
                    indices = active_batches[batch_index]
                    items = [ds['train'][index] for index in indices]
                    images = torch.stack([item[0] for item in items]).to(args.device)
                    masks = torch.stack([item[1] for item in items]).to(args.device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    if logits.shape != masks.shape:
                        raise ValueError(f'Output/target mismatch: {tuple(logits.shape)} != {tuple(masks.shape)}')
                    base_loss = shared.loss_fn(logits, masks)
                    total_loss = base_loss
                    if prototype is not None:
                        aux_loss = _prototype_auxiliary_loss(model, images, masks, prototype, shared.loss_fn)
                        total_loss = base_loss + LAMBDA_PROTOTYPE * aux_loss
                    if not torch.isfinite(total_loss):
                        raise ValueError('Nonfinite training loss')
                    value = float(total_loss.detach().item())
                    total_loss.backward()
                    optimizer.step()
                    train_loss_sum += value * len(indices)
                    train_case_count += len(indices)
                    next_batch_index = batch_index + 1
                    if time.perf_counter() - last_checkpoint_at >= CHECKPOINT_SECONDS:
                        checkpoint(status='running')
                    reason = should_pause()
                    if reason:
                        pause(reason)
                if train_case_count != len(ds['train']):
                    raise ValueError('Saved batch order did not consume every clinical training case exactly once')
                phase = 'val'
                val_batch_index = 0
                val_loss_sum = 0.0
                val_case_count = 0
                checkpoint(status='running')
            if phase == 'val':
                model.eval()
                n_val_batches = (len(ds['val']) + BATCH_SIZE - 1) // BATCH_SIZE
                with torch.inference_mode():
                    for val_index in range(val_batch_index, n_val_batches):
                        reason = should_pause()
                        if reason:
                            pause(reason)
                        start = val_index * BATCH_SIZE
                        items = [ds['val'][index] for index in range(start, min(start + BATCH_SIZE, len(ds['val'])))]
                        images = torch.stack([item[0] for item in items]).to(args.device)
                        masks = torch.stack([item[1] for item in items]).to(args.device)
                        logits = model(images)
                        if logits.shape != masks.shape:
                            raise ValueError('Validation output/target shape mismatch')
                        loss = shared.loss_fn(logits, masks)
                        if not torch.isfinite(loss):
                            raise ValueError('Nonfinite validation loss')
                        val_loss_sum += float(loss.item()) * len(items)
                        val_case_count += len(items)
                        val_batch_index = val_index + 1
                        if time.perf_counter() - last_checkpoint_at >= CHECKPOINT_SECONDS:
                            checkpoint(status='running')
                if val_case_count != len(ds['val']):
                    raise ValueError('Validation did not consume every clinical validation case')
                train_loss = train_loss_sum / max(1, train_case_count)
                val_loss = val_loss_sum / max(1, val_case_count)
                elapsed = epoch_elapsed_base + (time.perf_counter() - epoch_started)
                improved = val_loss < best_val - MIN_DELTA
                if improved:
                    best_val = val_loss
                    best_epoch = int(active_epoch)
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                history.append({'epoch': int(active_epoch), 'train_loss': train_loss, 'val_loss': val_loss, 'seconds': elapsed, 'best': bool(improved), 'best_epoch': best_epoch, 'bad_epochs': bad_epochs, 'lr': LEARNING_RATE})
                done_epoch = int(active_epoch)
                stop_early = bad_epochs >= PATIENCE
                done_budget = done_epoch >= MAX_EPOCHS
                if stop_early or done_budget:
                    active_epoch = None
                    phase = 'complete'
                    status = 'early_stopped' if stop_early else 'complete'
                else:
                    active_epoch += 1
                    phase = 'train'
                    active_batches = None
                    epoch_elapsed_base = 0.0
                    epoch_started = time.perf_counter()
                    next_batch_index = 0
                    train_loss_sum = 0.0
                    train_case_count = 0
                    val_batch_index = 0
                    val_loss_sum = 0.0
                    val_case_count = 0
                    status = 'running'
                checkpoint(status=status, save_best=improved)
                helpers._atomic_history(out / 'history.csv', history)
                row = history[-1]
                print(json.dumps(row, ensure_ascii=False), flush=True)
                if active_epoch is None:
                    atomic_json(out / 'completed.json', {'epochs_requested': MAX_EPOCHS, 'actual_epochs': len(history), 'best_epoch': best_epoch, 'best_val_loss': best_val, 'reason': 'early_stopped' if stop_early else 'completed', 'patience': PATIENCE, 'min_delta': MIN_DELTA, 'inputs_unchanged': True, 'variant': args.variant})
                    if (out / 'interrupted.json').exists():
                        (out / 'interrupted.json').unlink()
                    break
        shared.check_lock(data, lock)
        if public_manifest:
            final_public = _read_public_manifest(public_data / 'images', public_data / 'masks')
            if final_public['dataset_fingerprint'] != public_manifest['dataset_fingerprint']:
                raise RuntimeError('Public prototype support files changed during training')
        print(f'Training complete: {out}', flush=True)
        return out
    except KeyboardInterrupt as exc:
        if pause_state['reason'] is not None:
            print(str(exc), flush=True)
            return out
        checkpoint(status='paused')
        atomic_json(out / 'interrupted.json', {'status': 'paused', 'reason': 'keyboard_interrupt', 'active_epoch': active_epoch, 'completed_epochs': len(history), 'last_checkpoint': 'last.pt', 'last_checkpoint_sha256': sha256(out / 'last.pt'), 'saved_at': time.time()})
        return out
    finally:
        for sig, handler in previous_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

def evaluate(args) -> Path:
    import benchmark as shared
    variant = args.variant
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant: {variant}')
    original_project_modules = shared.project_modules

    def project_modules_for_variant():
        _, _, dataset_class, train_transform, eval_transform, imread, unpad = original_project_modules()

        def build(name, encoder_weights=None, transunet_root=None, transunet_pretrained=None):
            if name != MODEL_PREFIX + variant:
                raise ValueError(f'Unexpected evaluation model name: {name}')
            return _build_model(variant)

        def spec(name, encoder_weights=None, transunet_root=None, transunet_pretrained=None):
            if name != MODEL_PREFIX + variant:
                raise ValueError(f'Unexpected evaluation model name: {name}')
            return _model_spec(variant)
        return (build, spec, dataset_class, train_transform, eval_transform, imread, unpad)
    shared.project_modules = project_modules_for_variant
    try:
        namespace = SimpleNamespace(model=MODEL_PREFIX + variant, device=args.device, seed=SEED, encoder_weights=None, transunet_root=None, transunet_pretrained=None, data=Path(args.data), lock=Path(args.lock), out=Path(args.out), checkpoint=Path(args.checkpoint), split=args.split, probabilities=bool(args.probabilities), match_csv=None, torch_threads=TORCH_THREADS)
        shared.evaluate(namespace)
        return Path(args.out).resolve()
    finally:
        shared.project_modules = original_project_modules

def smoke(args) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    shared, _ = _runtime()
    torch.set_num_threads(TORCH_THREADS)
    shared.seed_all(SEED)
    model = _build_model(args.variant).to(args.device).train()
    images = torch.rand(2, 3, 256, 256, device=args.device)
    masks = (torch.rand(2, 1, 256, 256, device=args.device) > 0.95).float()
    logits = model(images)
    if logits.shape != masks.shape:
        raise AssertionError(f'Shape mismatch: {tuple(logits.shape)} / {tuple(masks.shape)}')
    base = shared.loss_fn(logits, masks)
    prototype_shape = None
    total = base
    if args.variant in ('prototype_only', 'full', 'full_no_augmentation'):
        with torch.no_grad():
            support_feature = model.unet.encoder(images)[-1]
            support_masks = F.interpolate(masks, size=support_feature.shape[-2:], mode='bilinear', align_corners=False)
            prototype = ((support_feature * (support_masks > 0.5)).sum((0, 2, 3)) / ((support_masks > 0.5).sum() + 1e-06)).view(1, -1, 1, 1)
        auxiliary = _prototype_auxiliary_loss(model, images, masks, prototype, shared.loss_fn)
        total = base + LAMBDA_PROTOTYPE * auxiliary
        prototype_shape = list(prototype.shape)
    if not torch.isfinite(total):
        raise AssertionError('Synthetic loss was not finite')
    total.backward()
    finite_gradients = all((torch.isfinite(parameter.grad).all().item() for parameter in model.parameters() if parameter.grad is not None))
    if not finite_gradients:
        raise AssertionError('Synthetic model produced nonfinite gradients')
    result = {'variant': args.variant, 'shape': list(logits.shape), 'loss': float(total.item()), 'prototype_shape': prototype_shape, 'finite_gradients': finite_gradients, 'real_training': False}
    print(json.dumps(result, indent=2), flush=True)
    return result

def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train_parser = sub.add_parser('train', help='Train one new component ablation')
    train_parser.add_argument('--variant', choices=VARIANTS, required=True)
    train_parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    train_parser.add_argument('--lock', type=Path, default=DEFAULT_LOCK)
    train_parser.add_argument('--out', type=Path, required=True)
    train_parser.add_argument('--public-data', type=Path, default=DEFAULT_PUBLIC)
    train_parser.add_argument('--device', default='cuda')
    train_parser.add_argument('--resume', action='store_true')
    eval_parser = sub.add_parser('evaluate', help='Export full benchmark evidence for one variant')
    eval_parser.add_argument('--variant', choices=VARIANTS, required=True)
    eval_parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    eval_parser.add_argument('--lock', type=Path, default=DEFAULT_LOCK)
    eval_parser.add_argument('--checkpoint', type=Path, required=True)
    eval_parser.add_argument('--out', type=Path, required=True)
    eval_parser.add_argument('--split', choices=('val', 'test'), default='test')
    eval_parser.add_argument('--device', default='cuda')
    eval_parser.add_argument('--probabilities', action='store_true')
    smoke_parser = sub.add_parser('smoke', help='Synthetic model/loss gradient check only')
    smoke_parser.add_argument('--variant', choices=VARIANTS, required=True)
    smoke_parser.add_argument('--device', default='cpu')
    return parser

def main() -> None:
    args = _parser().parse_args()
    if args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    else:
        smoke(args)
if __name__ == '__main__':
    main()
