from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import random
import signal
import tempfile
import time
from typing import Any
CHECKPOINT_FORMAT = 2

class TrainingPaused(RuntimeError):
    pass

def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

def _atomic_bytes(path: Path, write) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        if os.name != 'nt':
            try:
                dir_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise

def _atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    _atomic_bytes(path, lambda stream: stream.write(payload))

def _atomic_torch_save(torch, path: Path, value: Any) -> None:
    _atomic_bytes(path, lambda stream: torch.save(value, stream))

def _atomic_history(path: Path, history: list[dict[str, Any]]) -> None:
    fields = ['epoch', 'train_loss', 'val_loss', 'seconds', 'best', 'best_epoch', 'bad_epochs', 'lr']

    def write(stream):
        import io
        text = io.TextIOWrapper(stream, encoding='utf-8-sig', newline='', write_through=True)
        writer = csv.DictWriter(text, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
        text.flush()
        text.detach()
    _atomic_bytes(path, write)

def _pack_rng(value: Any, np) -> Any:
    if isinstance(value, np.ndarray):
        return {'__rng_type__': 'ndarray', 'dtype': str(value.dtype), 'shape': list(value.shape), 'values': value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {'__rng_type__': 'tuple', 'items': [_pack_rng(x, np) for x in value]}
    if isinstance(value, list):
        return [_pack_rng(x, np) for x in value]
    if isinstance(value, dict):
        return {str(k): _pack_rng(v, np) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unsupported RNG state value: {type(value).__name__}')

def _unpack_rng(value: Any, np) -> Any:
    if isinstance(value, dict) and value.get('__rng_type__') == 'tuple':
        return tuple((_unpack_rng(x, np) for x in value['items']))
    if isinstance(value, dict) and value.get('__rng_type__') == 'ndarray':
        arr = np.asarray(value['values'], dtype=value['dtype'])
        return arr.reshape(tuple(value['shape']))
    if isinstance(value, dict):
        return {k: _unpack_rng(v, np) for k, v in value.items()}
    if isinstance(value, list):
        return [_unpack_rng(x, np) for x in value]
    return value

def _augmentation_nodes(root):
    if root is None:
        return
    seen = set()
    stack = [('root', root)]
    while stack:
        path, node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        yield (path, node)
        children = getattr(node, 'transforms', None)
        if isinstance(children, (list, tuple)):
            for index in range(len(children) - 1, -1, -1):
                stack.append((f'{path}.transforms[{index}]', children[index]))

def _capture_augmentation_rng(train_dataset, np) -> list[dict[str, Any]]:
    transform = getattr(train_dataset, 'transform', None)
    augment = getattr(transform, 'augment', None)
    result = []
    for path, node in _augmentation_nodes(augment):
        item = {'path': path, 'class': f'{type(node).__module__}.{type(node).__qualname__}'}
        generator = getattr(node, 'random_generator', None)
        if generator is not None and hasattr(generator, 'bit_generator'):
            item['bit_generator_state'] = _pack_rng(generator.bit_generator.state, np)
        py_random = getattr(node, 'py_random', None)
        if isinstance(py_random, random.Random):
            item['py_random_state'] = _pack_rng(py_random.getstate(), np)
        if len(item) > 2:
            result.append(item)
    return result

def _restore_augmentation_rng(train_dataset, states, np) -> None:
    transform = getattr(train_dataset, 'transform', None)
    augment = getattr(transform, 'augment', None)
    nodes = {path: node for path, node in _augmentation_nodes(augment)}
    for item in states:
        node = nodes.get(item['path'])
        if node is None:
            raise ValueError(f"Training augmentation structure changed at {item['path']}")
        actual_class = f'{type(node).__module__}.{type(node).__qualname__}'
        if actual_class != item['class']:
            raise ValueError(f"Training augmentation class changed at {item['path']}")
        if 'bit_generator_state' in item:
            generator = getattr(node, 'random_generator', None)
            if generator is None or not hasattr(generator, 'bit_generator'):
                raise ValueError(f"Missing NumPy augmentation generator at {item['path']}")
            generator.bit_generator.state = _unpack_rng(item['bit_generator_state'], np)
        if 'py_random_state' in item:
            py_random = getattr(node, 'py_random', None)
            if not isinstance(py_random, random.Random):
                raise ValueError(f"Missing Python augmentation generator at {item['path']}")
            py_random.setstate(_unpack_rng(item['py_random_state'], np))

def _capture_rng(torch, np, sampler_generator, train_dataset) -> dict[str, Any]:
    state = {'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], 'python': _pack_rng(random.getstate(), np), 'numpy': _pack_rng(np.random.get_state(), np), 'sampler': sampler_generator.get_state(), 'albumentations': _capture_augmentation_rng(train_dataset, np)}
    return state

def _restore_rng(torch, np, sampler_generator, train_dataset, state) -> None:
    if torch.cuda.is_available() and state.get('torch_cuda'):
        saved_cuda = state['torch_cuda']
        if len(saved_cuda) != torch.cuda.device_count():
            raise ValueError('CUDA device count differs from the saved training run')
        torch.cuda.set_rng_state_all(saved_cuda)
    torch.set_rng_state(state['torch_cpu'])
    random.setstate(_unpack_rng(state['python'], np))
    np.random.set_state(_unpack_rng(state['numpy'], np))
    sampler_generator.set_state(state['sampler'])
    _restore_augmentation_rng(train_dataset, state.get('albumentations', []), np)

def _make_batches(shared, size: int, batch_size: int, generator) -> list[list[int]]:
    return [list(map(int, batch)) for batch in shared.CompleteBatchSampler(size, batch_size, generator)]

class _Progress:

    def __init__(self, total: int, initial: int, description: str):
        self._bar = None
        self._done = initial
        self._total = total
        try:
            from tqdm.auto import tqdm
            self._bar = tqdm(total=total, initial=initial, desc=description, leave=False, dynamic_ncols=True, mininterval=0.2, ascii=True)
        except Exception:
            print(f'{description}: {initial}/{total} batches', flush=True)

    def update(self, loss: float | None=None, lr: float | None=None) -> None:
        self._done += 1
        if self._bar is not None:
            self._bar.update(1)
            values = {}
            if loss is not None:
                values['loss'] = f'{loss:.5f}'
            if lr is not None:
                values['lr'] = f'{lr:.2g}'
            if values:
                self._bar.set_postfix(values)
        else:
            print(f'  batch {self._done}/{self._total}' + (f' loss={loss:.6f}' if loss is not None else ''), flush=True)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()

class _LiveCurve:

    def __init__(self, out: Path, enabled: bool):
        self.out = out
        self.enabled = enabled
        self.plt = None
        self.fig = None
        self.ax = None
        self.warning_printed = False
        try:
            import matplotlib
            if not enabled:
                matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            self.plt = plt
            if enabled:
                plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 5))
            self.fig.canvas.manager.set_window_title('Segmentation training loss')
            if enabled:
                plt.show(block=False)
        except Exception as exc:
            self._warn(exc)
            self.plt = None

    def _warn(self, exc) -> None:
        if not self.warning_printed:
            print(f'Curve display/save unavailable: {type(exc).__name__}: {exc}', flush=True)
            self.warning_printed = True

    def update(self, history, active_epoch: int | None=None, batch_index: int=0, batch_total: int=1, running_train_loss: float | None=None, save: bool=False) -> None:
        if self.plt is None or self.ax is None:
            return
        try:
            self.ax.clear()
            epochs = [row['epoch'] for row in history]
            train = [row['train_loss'] for row in history]
            valid = [row['val_loss'] for row in history]
            if epochs:
                self.ax.plot(epochs, train, 'o-', label='train')
                self.ax.plot(epochs, valid, 'o-', label='validation')
            if active_epoch is not None and running_train_loss is not None:
                progress_x = active_epoch - 1 + batch_index / max(1, batch_total)
                self.ax.plot([active_epoch - 1, progress_x], [running_train_loss, running_train_loss], '--', color='tab:blue', alpha=0.65, label='current train')
            self.ax.set_xlabel('Epoch')
            self.ax.set_ylabel('Loss')
            self.ax.set_title('Training and validation loss')
            self.ax.grid(True, alpha=0.25)
            if epochs or active_epoch is not None:
                self.ax.legend(loc='best')
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            if self.enabled:
                self.plt.pause(0.001)
            if save:
                fd, tmp_name = tempfile.mkstemp(prefix='training_curves.', suffix='.tmp.png', dir=self.out)
                os.close(fd)
                tmp = Path(tmp_name)
                try:
                    self.fig.savefig(tmp, format='png', dpi=130)
                    with tmp.open('r+b') as stream:
                        os.fsync(stream.fileno())
                    os.replace(tmp, self.out / 'training_curves.png')
                finally:
                    tmp.unlink(missing_ok=True)
        except Exception as exc:
            self._warn(exc)
            self.plt = None

def train(args):
    import benchmark as shared
    import numpy as np
    import torch
    if args.model == 'retained':
        raise ValueError('retained is evaluation-only; original prototype training is unchanged')
    epochs = int(getattr(args, 'epochs', 100))
    batch_size = int(getattr(args, 'batch_size', 4))
    lr = float(getattr(args, 'lr', 0.0001))
    patience = int(getattr(args, 'patience', 5))
    min_delta = float(getattr(args, 'min_delta', 0.0))
    checkpoint_seconds = float(getattr(args, 'checkpoint_seconds', 120.0))
    torch_threads = int(getattr(args, 'torch_threads', 4))
    resume = bool(getattr(args, 'resume', False))
    show_plot = bool(getattr(args, 'show_plot', False))
    if epochs <= 0 or batch_size <= 0 or lr <= 0:
        raise ValueError('epochs, batch size, and lr must be positive')
    if patience <= 0 or min_delta < 0 or checkpoint_seconds <= 0 or (torch_threads <= 0):
        raise ValueError('patience, checkpoint interval, and torch_threads must be positive; min_delta must be nonnegative')
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(max(1, min(torch_threads, 4)))
    except RuntimeError:
        pass
    out = Path(args.out).expanduser().resolve()
    if resume:
        if not out.is_dir():
            raise FileNotFoundError(f'Resume run directory does not exist: {out}')
        if not (out / 'last.pt').is_file():
            raise FileNotFoundError(f'No resumable last.pt checkpoint in {out}')
        if (out / 'completed.json').exists():
            raise ValueError('This run is already complete; choose a new output directory for another run')
    elif out.exists():
        raise FileExistsError(f'Output already exists; choose a new run directory: {out}')
    frozen = shared.check_lock(args.data, args.lock)
    shared.seed_all(args.seed)
    build, spec, *_ = shared.project_modules()
    model_kwargs = shared.model_kwargs(args, True)
    model_spec = _jsonable(spec(args.model, **model_kwargs))
    source_hashes = {'benchmark_sha256': shared.sha(shared.__file__), 'training_sha256': shared.sha(__file__), 'factory_sha256': shared.sha(shared.HOME / 'models.py'), 'shared_code_sha256': frozen['shared_code_sha256']}
    transunet_root = getattr(args, 'transunet_root', None)
    transunet_pretrained = getattr(args, 'transunet_pretrained', None)
    pretrained_sha = None
    if transunet_pretrained:
        pretrained_path = Path(transunet_pretrained)
        if not pretrained_path.is_file():
            raise FileNotFoundError(f'TransUNet pretrained file does not exist: {pretrained_path}')
        pretrained_sha = shared.sha(pretrained_path)
    if args.encoder_weights:
        init_file = Path(torch.hub.get_dir()) / 'checkpoints/resnet34-b627a593.pth'
        if not init_file.is_file():
            raise FileNotFoundError('Expected ImageNet ResNet34 initialization cache is missing')
        source_hashes['encoder_pretrained_sha256'] = shared.sha(init_file)
    if pretrained_sha:
        source_hashes['transunet_pretrained_sha256'] = pretrained_sha
    signature = {'model': args.model, 'device': str(args.device), 'seed': int(args.seed), 'batch_size': batch_size, 'lr': lr, 'patience': patience, 'min_delta': min_delta, 'encoder_weights': getattr(args, 'encoder_weights', None), 'transunet_root': str(Path(transunet_root).resolve()) if transunet_root else None, 'transunet_pretrained': str(Path(transunet_pretrained).resolve()) if transunet_pretrained else None, 'dataset_fingerprint': frozen['dataset_fingerprint'], 'protocol': _jsonable(shared.PROTOCOL), 'model_spec': model_spec, 'source_hashes': source_hashes}
    saved_config = None
    saved_state = None
    if resume:
        config_path = out / 'config.json'
        lock_path = out / 'dataset_lock.json'
        if not config_path.is_file() or not lock_path.is_file():
            raise FileNotFoundError('Resume directory is missing config.json or dataset_lock.json')
        saved_config = json.loads(config_path.read_text(encoding='utf-8'))
        if saved_config.get('training_signature') != signature:
            raise ValueError('Resume configuration, dataset, model, protocol, or source hashes do not match')
        saved_lock = json.loads(lock_path.read_text(encoding='utf-8'))
        if saved_lock.get('dataset_fingerprint') != frozen['dataset_fingerprint']:
            raise ValueError('Saved run dataset lock differs from the current frozen dataset')
        saved_state = torch.load(out / 'last.pt', map_location='cpu', weights_only=True)
        if saved_state.get('checkpoint_format') != CHECKPOINT_FORMAT:
            raise ValueError('Unsupported or incomplete resume checkpoint format')
        if saved_state.get('training_signature') != signature:
            raise ValueError('Checkpoint training signature does not match this run')
        progress_epoch = saved_state.get('active_epoch') or saved_state.get('completed_epochs', 0)
        if epochs < progress_epoch:
            raise ValueError(f'Requested epoch cap {epochs} is below saved progress {progress_epoch}')
        prior_cap = int(saved_config.get('max_epochs', epochs))
        if epochs < prior_cap:
            raise ValueError(f'Resume may extend, but cannot reduce, the saved epoch cap ({prior_cap})')
    ds = shared.datasets(args.data)
    train_augment = getattr(ds['train'].transform, 'augment', None)
    if train_augment is not None and hasattr(train_augment, 'set_random_seed'):
        train_augment.set_random_seed(args.seed)
    if len(ds['train']) < 2 or len(ds['val']) < 1:
        raise ValueError('Training requires at least two training cases and one validation case')
    sampler_generator = torch.Generator().manual_seed(args.seed)
    model = build(args.model, **model_kwargs).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if resume:
        model.load_state_dict(saved_state.get('model_state', saved_state.get('state_dict')), strict=True)
        optimizer.load_state_dict(saved_state['optimizer_state'])
        _restore_rng(torch, np, sampler_generator, ds['train'], saved_state['rng_state'])
        history = list(saved_state['history'])
        best_val = float(saved_state['best_val_loss'])
        best_epoch = int(saved_state['best_epoch'])
        bad_epochs = int(saved_state['bad_epochs'])
        active_epoch = saved_state.get('active_epoch')
        active_batches = saved_state.get('active_batches')
        next_batch_index = int(saved_state.get('next_batch_index', 0))
        train_loss_sum = float(saved_state.get('train_loss_sum', 0.0))
        train_case_count = int(saved_state.get('train_case_count', 0))
        val_batch_index = int(saved_state.get('val_batch_index', 0))
        val_loss_sum = float(saved_state.get('val_loss_sum', 0.0))
        val_case_count = int(saved_state.get('val_case_count', 0))
        elapsed_base = float(saved_state.get('elapsed_seconds', 0.0))
        epoch_elapsed_base = float(saved_state.get('active_epoch_elapsed_seconds', 0.0))
        run_state = str(saved_state.get('status', 'running'))
        (out / 'interrupted.json').unlink(missing_ok=True)
        saved_config['max_epochs'] = epochs
        saved_config['resume_count'] = int(saved_config.get('resume_count', 0)) + 1
        saved_config['last_resumed_at'] = time.time()
        _atomic_json(out / 'config.json', saved_config)
        _atomic_history(out / 'history.csv', history)
    else:
        shared.new_output(out)
        history = []
        best_val = float('inf')
        best_epoch = 0
        bad_epochs = 0
        active_epoch = 1
        active_batches = _make_batches(shared, len(ds['train']), batch_size, sampler_generator)
        next_batch_index = 0
        train_loss_sum = 0.0
        train_case_count = 0
        val_batch_index = 0
        val_loss_sum = 0.0
        val_case_count = 0
        elapsed_base = 0.0
        epoch_elapsed_base = 0.0
        run_state = 'running'
        raw_args = {k: _jsonable(v) for k, v in vars(args).items()}
        config = {'arguments': raw_args, 'training_signature': signature, 'dataset_fingerprint': frozen['dataset_fingerprint'], 'protocol': _jsonable(shared.PROTOCOL), 'runtime': shared.runtime(), 'model_spec': model_spec, 'source_hashes': source_hashes, 'max_epochs': epochs, 'patience': patience, 'min_delta': min_delta, 'checkpoint_seconds': checkpoint_seconds, 'torch_threads': torch_threads, 'training_policy': 'Adam; full train split once per epoch; early stopping on validation loss', 'training_batch_policy': 'CompleteBatchSampler; every training case once per epoch; no workers', 'external_clinical_support': False, 'resume_count': 0, 'created_at': time.time()}
        _atomic_json(out / 'config.json', config)
        _atomic_json(out / 'dataset_lock.json', frozen)
        saved_config = config
    stop_arg = getattr(args, 'stop_file', None)
    stop_file = Path(stop_arg).expanduser().resolve() if stop_arg else out / 'STOP_REQUESTED'
    curve = _LiveCurve(out, show_plot)
    if resume:
        _restore_rng(torch, np, sampler_generator, ds['train'], saved_state['rng_state'])
    process_started = time.perf_counter()
    epoch_segment_started = process_started
    last_checkpoint_at = process_started
    last_progress_at = 0.0
    step_in_progress = False
    interrupted_by = {'signal': None}
    original_handlers = {}

    def signal_handler(signum, _frame):
        interrupted_by['signal'] = signum
    for sig_name in ('SIGINT', 'SIGTERM', 'SIGBREAK'):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            original_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, signal_handler)
        except (ValueError, OSError):
            original_handlers.pop(sig, None)

    def total_elapsed(now=None):
        return elapsed_base + ((time.perf_counter() if now is None else now) - process_started)

    def active_epoch_elapsed(now=None):
        return epoch_elapsed_base + ((time.perf_counter() if now is None else now) - epoch_segment_started)

    def current_lr():
        return float(optimizer.param_groups[0]['lr'])

    def save_checkpoint(status=None, phase=None, save_best=False):
        nonlocal elapsed_base, epoch_elapsed_base, process_started, epoch_segment_started, last_checkpoint_at
        now = time.perf_counter()
        elapsed_value = total_elapsed(now)
        epoch_value = active_epoch_elapsed(now) if active_epoch is not None else 0.0
        model_state = model.state_dict()
        checkpoint = {'checkpoint_format': CHECKPOINT_FORMAT, 'model': args.model, 'seed': int(args.seed), 'dataset_fingerprint': frozen['dataset_fingerprint'], 'protocol': _jsonable(shared.PROTOCOL), 'training_signature': signature, 'model_state': model_state, 'state_dict': model_state, 'optimizer_state': optimizer.state_dict(), 'active_epoch': active_epoch, 'active_batches': active_batches, 'next_batch_index': next_batch_index, 'train_loss_sum': train_loss_sum, 'train_case_count': train_case_count, 'val_batch_index': val_batch_index, 'val_loss_sum': val_loss_sum, 'val_case_count': val_case_count, 'completed_epochs': len(history), 'history': history, 'best_val_loss': best_val, 'best_epoch': best_epoch, 'bad_epochs': bad_epochs, 'rng_state': _capture_rng(torch, np, sampler_generator, ds['train']), 'elapsed_seconds': elapsed_value, 'active_epoch_elapsed_seconds': epoch_value, 'phase': phase or ('train' if active_epoch is not None else 'complete'), 'status': status or 'running', 'max_epochs': epochs}
        if save_best:
            _atomic_torch_save(torch, out / 'best.pt', checkpoint)
        _atomic_torch_save(torch, out / 'last.pt', checkpoint)
        elapsed_base = elapsed_value
        epoch_elapsed_base = epoch_value
        process_started = time.perf_counter()
        epoch_segment_started = process_started
        last_checkpoint_at = process_started

    def make_progress(phase: str, batch: int=0, total_batches: int=0, train_value: float | None=None, val_value: float | None=None, status: str='running') -> dict[str, Any]:
        now = time.perf_counter()
        elapsed = total_elapsed(now)
        if active_epoch is None:
            eta = 0.0
        elif phase == 'train' and total_batches:
            completed_in_epoch = max(1, batch)
            seconds_per_batch = max(0.0, active_epoch_elapsed(now)) / completed_in_epoch
            remaining_current = max(0, total_batches - batch) * seconds_per_batch
            if history:
                avg_epoch = sum((float(row['seconds']) for row in history)) / len(history)
                future_epochs = max(0, epochs - active_epoch)
                eta = remaining_current + future_epochs * avg_epoch
            else:
                eta = remaining_current
        else:
            eta = max(0.0, active_epoch_elapsed(now))
        return {'status': status, 'phase': phase, 'epoch': active_epoch, 'epochs': epochs, 'batch': batch, 'batches': total_batches, 'train_loss': train_value, 'val_loss': val_value, 'best_val_loss': best_val if best_val != float('inf') else None, 'best_epoch': best_epoch, 'bad_epochs': bad_epochs, 'patience': patience, 'learning_rate': current_lr(), 'elapsed_seconds': elapsed, 'eta_seconds': eta, 'stop_file': str(stop_file), 'last_checkpoint': str(out / 'last.pt')}

    def write_progress(phase: str, **kwargs):
        progress = make_progress(phase, **kwargs)
        _atomic_json(out / 'progress.json', progress)
        record = {**progress, 'recorded_at': time.time(), 'resume_attempt': saved_config.get('resume_count', 0)}
        with (out / 'progress_history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def check_stop() -> str | None:
        if interrupted_by['signal'] is not None:
            try:
                return f"signal:{signal.Signals(interrupted_by['signal']).name}"
            except Exception:
                return f"signal:{interrupted_by['signal']}"
        if stop_file.exists():
            return 'stop_file'
        return None

    def pause(reason: str, phase: str):
        save_checkpoint(status='paused', phase=phase)
        marker = {'status': 'paused', 'reason': reason, 'active_epoch': active_epoch, 'completed_epochs': len(history), 'last_checkpoint': 'last.pt', 'last_checkpoint_sha256': shared.sha(out / 'last.pt'), 'saved_at': time.time()}
        _atomic_json(out / 'interrupted.json', marker)
        if stop_file.exists():
            try:
                stop_file.unlink()
            except OSError:
                pass
        write_progress(phase, status='paused')
        print(f'Training paused ({reason}); resume with --resume from {out}', flush=True)
        raise TrainingPaused(f'Training paused; resumable checkpoint saved in {out}')

    def update_plot_and_progress(phase, batch, total_batches, train_value=None, val_value=None, batch_total_for_curve=1):
        nonlocal last_progress_at
        now = time.perf_counter()
        should_flush = now - last_progress_at >= 5.0
        if should_flush or curve.enabled:
            curve.update(history, active_epoch, batch, batch_total_for_curve, train_value, save=should_flush)
        if should_flush:
            write_progress(phase, batch=batch, total_batches=total_batches, train_value=train_value, val_value=val_value)
            last_progress_at = now
    try:
        if not resume:
            save_checkpoint(status='running', phase='train')
            _atomic_history(out / 'history.csv', history)
            write_progress('train', batch=0, total_batches=len(active_batches), status='running')
        else:
            write_progress(str(saved_state.get('phase', 'train')), batch=next_batch_index, total_batches=len(active_batches or []), status='running')
        if resume and active_epoch is None and (saved_state.get('status') in ('complete', 'early_stopped')):
            shared.check_lock(args.data, args.lock)
            actual_epoch = len(history)
            reason = 'early_stopped' if saved_state['status'] == 'early_stopped' else 'completed'
            _atomic_json(out / 'completed.json', {'epochs_requested': epochs, 'actual_epochs': actual_epoch, 'best_epoch': best_epoch, 'best_val_loss': best_val, 'reason': reason, 'patience': patience, 'min_delta': min_delta, 'inputs_unchanged': True})
            (out / 'interrupted.json').unlink(missing_ok=True)
            write_progress('complete', status='complete')
            return out
        while active_epoch is not None and active_epoch <= epochs:
            if active_batches is None:
                active_batches = _make_batches(shared, len(ds['train']), batch_size, sampler_generator)
                next_batch_index = 0
                train_loss_sum = 0.0
                train_case_count = 0
                val_batch_index = 0
                val_loss_sum = 0.0
                val_case_count = 0
                epoch_elapsed_base = 0.0
                epoch_segment_started = time.perf_counter()
            n_train_batches = len(active_batches)
            if not 0 <= next_batch_index <= n_train_batches:
                raise ValueError('Checkpoint train batch index is outside the saved epoch order')
            model.train()
            pbar = _Progress(n_train_batches, next_batch_index, f'Epoch {active_epoch}/{epochs} train')
            try:
                for batch_index in range(next_batch_index, n_train_batches):
                    reason = check_stop()
                    if reason:
                        pause(reason, 'train')
                    indices = active_batches[batch_index]
                    step_in_progress = True
                    items = [ds['train'][index] for index in indices]
                    x = torch.stack([item[0] for item in items]).to(args.device)
                    y = torch.stack([item[1] for item in items]).to(args.device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    if logits.shape != y.shape:
                        raise ValueError(f'Output/target mismatch: {logits.shape}, {y.shape}')
                    loss = shared.loss_fn(logits, y)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite training loss')
                    value = float(loss.detach().item())
                    loss.backward()
                    step_in_progress = True
                    optimizer.step()
                    train_loss_sum += value * len(indices)
                    train_case_count += len(indices)
                    next_batch_index = batch_index + 1
                    step_in_progress = False
                    pbar.update(train_loss_sum / max(1, train_case_count), current_lr())
                    running = train_loss_sum / max(1, train_case_count)
                    update_plot_and_progress('train', next_batch_index, n_train_batches, train_value=running, batch_total_for_curve=n_train_batches)
                    if time.perf_counter() - last_checkpoint_at >= checkpoint_seconds:
                        save_checkpoint(status='running', phase='train')
                        write_progress('train', batch=next_batch_index, total_batches=n_train_batches, train_value=running)
                    reason = check_stop()
                    if reason:
                        pause(reason, 'train')
            finally:
                pbar.close()
            if train_case_count != len(ds['train']):
                raise ValueError('Saved batch order did not consume every training case exactly once')
            reason = check_stop()
            if reason:
                pause(reason, 'val')
            n_val_batches = (len(ds['val']) + batch_size - 1) // batch_size
            if not 0 <= val_batch_index <= n_val_batches:
                raise ValueError('Checkpoint validation batch index is outside the validation split')
            model.eval()
            vbar = _Progress(n_val_batches, val_batch_index, f'Epoch {active_epoch}/{epochs} validation')
            try:
                with torch.inference_mode():
                    for val_index in range(val_batch_index, n_val_batches):
                        reason = check_stop()
                        if reason:
                            pause(reason, 'val')
                        start = val_index * batch_size
                        items = [ds['val'][index] for index in range(start, min(start + batch_size, len(ds['val'])))]
                        x = torch.stack([item[0] for item in items]).to(args.device)
                        y = torch.stack([item[1] for item in items]).to(args.device)
                        logits = model(x)
                        if logits.shape != y.shape:
                            raise ValueError(f'Output/target mismatch: {logits.shape}, {y.shape}')
                        loss = shared.loss_fn(logits, y)
                        if not torch.isfinite(loss):
                            raise ValueError('Nonfinite validation loss')
                        value = float(loss.item())
                        val_loss_sum += value * len(items)
                        val_case_count += len(items)
                        val_batch_index = val_index + 1
                        vbar.update(val_loss_sum / max(1, val_case_count), current_lr())
                        update_plot_and_progress('val', val_batch_index, n_val_batches, train_value=train_loss_sum / max(1, train_case_count), val_value=val_loss_sum / max(1, val_case_count), batch_total_for_curve=n_val_batches)
                        if time.perf_counter() - last_checkpoint_at >= checkpoint_seconds:
                            save_checkpoint(status='running', phase='val')
                            write_progress('val', batch=val_batch_index, total_batches=n_val_batches, train_value=train_loss_sum / max(1, train_case_count), val_value=val_loss_sum / max(1, val_case_count))
                        reason = check_stop()
                        if reason:
                            pause(reason, 'val')
            finally:
                vbar.close()
            if val_case_count != len(ds['val']):
                raise ValueError('Validation epoch did not consume the full validation split')
            train_value = train_loss_sum / train_case_count
            val_value = val_loss_sum / val_case_count
            improved = val_value < best_val - min_delta
            if improved:
                best_val = val_value
                best_epoch = int(active_epoch)
                bad_epochs = 0
            else:
                bad_epochs += 1
            epoch_seconds = active_epoch_elapsed()
            row = {'epoch': int(active_epoch), 'train_loss': train_value, 'val_loss': val_value, 'seconds': epoch_seconds, 'best': bool(improved), 'best_epoch': best_epoch, 'bad_epochs': bad_epochs, 'lr': current_lr()}
            history.append(row)
            actual_epoch = int(active_epoch)
            early_stopped = not improved and bad_epochs >= patience
            finished_budget = actual_epoch >= epochs
            if early_stopped or finished_budget:
                active_epoch = None
                active_batches = None
                next_batch_index = 0
                train_loss_sum = 0.0
                train_case_count = 0
                val_batch_index = 0
                val_loss_sum = 0.0
                val_case_count = 0
                end_status = 'early_stopped' if early_stopped else 'complete'
            else:
                active_epoch = actual_epoch + 1
                active_batches = _make_batches(shared, len(ds['train']), batch_size, sampler_generator)
                next_batch_index = 0
                train_loss_sum = 0.0
                train_case_count = 0
                val_batch_index = 0
                val_loss_sum = 0.0
                val_case_count = 0
                epoch_elapsed_base = 0.0
                epoch_segment_started = time.perf_counter()
                end_status = 'running'
            _atomic_history(out / 'history.csv', history)
            save_checkpoint(status=end_status, phase='complete' if active_epoch is None else 'train', save_best=improved)
            curve.update(history, active_epoch, 0, len(active_batches) if active_batches else 1, None, save=True)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            write_progress('complete' if active_epoch is None else 'train', batch=0 if active_epoch is None else next_batch_index, total_batches=0 if active_epoch is None else len(active_batches), train_value=train_value, val_value=val_value, status='complete' if active_epoch is None else 'running')
            if active_epoch is None:
                shared.check_lock(args.data, args.lock)
                reason = 'early_stopped' if early_stopped else 'completed'
                _atomic_json(out / 'completed.json', {'epochs_requested': epochs, 'actual_epochs': actual_epoch, 'best_epoch': best_epoch, 'best_val_loss': best_val, 'reason': reason, 'patience': patience, 'min_delta': min_delta, 'inputs_unchanged': True})
                (out / 'interrupted.json').unlink(missing_ok=True)
                write_progress('complete', status='complete', train_value=train_value, val_value=val_value)
                print(f'Training {reason}; best epoch {best_epoch}, validation loss {best_val:.6f}', flush=True)
                return out
        raise RuntimeError('Training exited without a completed or paused run state')
    except TrainingPaused:
        raise
    except BaseException as exc:
        checkpoint_saved = False
        if not step_in_progress:
            try:
                save_checkpoint(status='error', phase='train' if active_epoch is not None else 'complete')
                checkpoint_saved = True
            except BaseException:
                pass
        try:
            marker = {'status': 'error', 'error_type': type(exc).__name__, 'active_epoch': active_epoch, 'completed_epochs': len(history), 'last_checkpoint': 'last.pt' if checkpoint_saved or (out / 'last.pt').exists() else None, 'saved_at': time.time()}
            _atomic_json(out / 'interrupted.json', marker)
            if active_epoch is not None:
                write_progress('error', status='error')
        except BaseException:
            pass
        raise
    finally:
        for sig, handler in original_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
