from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

def _first(mapping: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None

def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None

def _display_number(value: Any, digits: int=4) -> str:
    number = _number(value)
    return f'{number:.{digits}f}' if number is not None else '—'

def _duration(value: Any) -> str | None:
    seconds = _number(value)
    if seconds is None or seconds < 0:
        return None
    whole = int(seconds)
    hours, remainder = divmod(whole, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

def _session_state_path(session_path: Path) -> Path:
    if session_path.name.lower() == 'session_state.json':
        return session_path
    if session_path.suffix.lower() == '.json' and (not session_path.is_dir()):
        return session_path
    return session_path / 'session_state.json'

def _read_json(path: Path, previous: dict[str, Any] | None, expected_keys: tuple[str, ...]) -> dict[str, Any] | None:
    try:
        with path.open('r', encoding='utf-8-sig') as stream:
            candidate = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return previous
    if not isinstance(candidate, dict) or not any((key in candidate for key in expected_keys)):
        return previous
    return candidate

def _csv_value(row: dict[str, Any], *names: str) -> Any:
    lower_keys = {key.strip().lower(): value for key, value in row.items() if key}
    for name in names:
        if name.lower() in lower_keys:
            return lower_keys[name.lower()]
    return None

def _read_history(path: Path, previous: list[dict[str, float]] | None) -> list[dict[str, float]] | None:
    try:
        text = path.read_text(encoding='utf-8-sig')
    except (OSError, UnicodeError):
        return previous
    if not text or not text.endswith(('\n', '\r')):
        return previous
    try:
        reader = csv.DictReader(text.splitlines())
        fields = {str(name).strip().lower() for name in reader.fieldnames or []}
        if not fields:
            return previous
        parsed: list[dict[str, float]] = []
        for row in reader:
            if not row or all((value is None or not str(value).strip() for value in row.values())):
                continue
            if None in row:
                return previous
            epoch = _number(_csv_value(row, 'epoch', 'current_epoch', 'epochs'))
            train = _number(_csv_value(row, 'train_loss', 'training_loss', 'loss'))
            val_raw = _csv_value(row, 'val_loss', 'validation_loss', 'valid_loss')
            val = _number(val_raw) if val_raw is not None and str(val_raw).strip() else None
            if epoch is None or train is None:
                return previous
            if {'val_loss', 'validation_loss', 'valid_loss'} & fields and val is None:
                return previous
            item = {'epoch': epoch, 'train_loss': train}
            if val is not None:
                item['val_loss'] = val
            parsed.append(item)
        return parsed
    except (csv.Error, TypeError, ValueError):
        return previous

def _progress_values(progress: dict[str, Any] | None) -> dict[str, Any]:
    if not progress:
        return {}
    return {'phase': _first(progress, 'phase', 'status', 'stage'), 'epoch': _number(_first(progress, 'epoch', 'current_epoch')), 'max_epochs': _number(_first(progress, 'max_epochs', 'epochs', 'total_epochs')), 'batch': _number(_first(progress, 'batch', 'current_batch', 'batch_index')), 'total_batches': _number(_first(progress, 'total_batches', 'batches', 'num_batches')), 'train_loss': _number(_first(progress, 'train_loss', 'training_loss')), 'val_loss': _number(_first(progress, 'val_loss', 'validation_loss', 'valid_loss')), 'loss': _number(_first(progress, 'loss', 'current_loss')), 'elapsed': _number(_first(progress, 'elapsed_seconds', 'elapsed', 'seconds_elapsed')), 'eta': _number(_first(progress, 'eta_seconds', 'eta', 'estimated_remaining_seconds')), 'best_val': _number(_first(progress, 'best_val', 'bestVal', 'best_val_loss', 'best_validation_loss'))}

def _plot_points(history: list[dict[str, float]] | None, progress: dict[str, Any] | None) -> tuple[list[float], list[float], list[float], list[float]]:
    train_x: list[float] = []
    train_y: list[float] = []
    val_x: list[float] = []
    val_y: list[float] = []
    points = list(history or [])
    values = _progress_values(progress)
    epoch = values.get('epoch')
    train = values.get('train_loss')
    val = values.get('val_loss')
    if train is None and values.get('loss') is not None and (str(values.get('phase') or '').lower() in {'train', 'training'}):
        train = values['loss']
    if epoch is not None and train is not None:
        if not points or points[-1].get('epoch') != epoch:
            points.append({'epoch': epoch, 'train_loss': train})
        else:
            points[-1] = {**points[-1], 'train_loss': train}
        if val is not None:
            points[-1]['val_loss'] = val
    for point in points:
        x = _number(point.get('epoch'))
        train_y_value = _number(point.get('train_loss'))
        val_y_value = _number(point.get('val_loss'))
        if x is None:
            continue
        if train_y_value is not None:
            train_x.append(x)
            train_y.append(train_y_value)
        if val_y_value is not None:
            val_x.append(x)
            val_y.append(val_y_value)
    return (train_x, train_y, val_x, val_y)

def _best_value(progress: dict[str, Any] | None, history: list[dict[str, float]] | None) -> float | None:
    best = _progress_values(progress).get('best_val')
    if best is not None:
        return best
    values = [_number(point.get('val_loss')) for point in history or []]
    finite = [value for value in values if value is not None]
    return min(finite) if finite else None

def _gui_backend_available(matplotlib: Any) -> bool:
    backend = str(matplotlib.get_backend())
    try:
        from matplotlib.backends.registry import backend_registry
        return bool(backend_registry.is_gui_backend(backend))
    except (ImportError, AttributeError):
        name = backend.lower()
        non_gui = {'agg', 'pdf', 'ps', 'svg', 'cairo', 'template', 'pgf'}
        if name in non_gui or name.startswith('module://matplotlib.backends.backend_'):
            return False
        return any((token in name for token in ('tk', 'qt', 'gtk', 'wx', 'macosx')))

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description='Read-only live training curves and progress viewer')
    parser.add_argument('--session', required=True, type=Path, help='session directory or path to session_state.json')
    args = parser.parse_args(argv)
    session_state_path = _session_state_path(args.session.expanduser())
    try:
        import matplotlib
    except ImportError as exc:
        print(f'Cannot start live viewer: Matplotlib is unavailable ({exc}).', file=sys.stderr)
        return 2
    if not _gui_backend_available(matplotlib):
        print(f"Cannot start live viewer: Matplotlib backend '{matplotlib.get_backend()}' has no GUI. Select an installed interactive backend and run again.", file=sys.stderr)
        return 2
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.widgets import Button
    except Exception as exc:
        print(f'Cannot start live viewer: the interactive Matplotlib backend failed to load ({exc}).', file=sys.stderr)
        return 2
    figure, (loss_axis, info_axis) = plt.subplots(1, 2, figsize=(11, 5.5), gridspec_kw={'width_ratios': [1.5, 1.0]})
    figure.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.18, wspace=0.28)
    info_axis.axis('off')
    info_text = info_axis.text(0.02, 0.98, 'Waiting for session state…', va='top', ha='left', family='monospace', fontsize=10, transform=info_axis.transAxes)
    button_axis = figure.add_axes([0.72, 0.045, 0.22, 0.075])
    pause_button = Button(button_axis, 'Save & Pause')
    cache: dict[str, Any] = {'session': None, 'progress': None, 'history': None, 'run_key': None, 'run_path': None, 'notice': ''}

    def refresh(_frame: int) -> None:
        cache['session'] = _read_json(session_state_path, cache['session'], ('status', 'current_model', 'current_run', 'models', 'run_id'))
        state = cache['session'] or {}
        model = _first(state, 'current_model', 'model')
        run_value = _first(state, 'current_run')
        run_path = Path(str(run_value)).expanduser() if run_value else None
        if run_path is not None and (not run_path.is_absolute()):
            run_path = session_state_path.parent / run_path
        run_key = (str(run_path) if run_path else None, str(model or ''), str(_first(state, 'run_id') or ''))
        if run_key != cache['run_key']:
            cache['run_key'] = run_key
            cache['run_path'] = run_path
            cache['progress'] = None
            cache['history'] = None
            cache['notice'] = ''
        else:
            cache['run_path'] = run_path
        if run_path is not None:
            cache['progress'] = _read_json(run_path / 'progress.json', cache['progress'], ('phase', 'epoch', 'current_epoch', 'batch', 'loss', 'train_loss', 'val_loss', 'elapsed_seconds', 'eta_seconds'))
            cache['history'] = _read_history(run_path / 'history.csv', cache['history'])
        progress = cache['progress']
        history = cache['history']
        values = _progress_values(progress)
        status = str(_first(state, 'status') or ('Waiting' if not state else 'Running'))
        model_label = str(model or '—')
        best = _best_value(progress, history)
        best_label = _display_number(best) if best is not None else '—'
        figure.suptitle(f'{status}  |  {model_label}  |  bestVal {best_label}', fontsize=13)
        loss_axis.clear()
        loss_axis.set_title('Loss by epoch')
        loss_axis.set_xlabel('Epoch')
        loss_axis.set_ylabel('Loss')
        train_x, train_y, val_x, val_y = _plot_points(history, progress)
        if train_x:
            loss_axis.plot(train_x, train_y, label='Training', color='#2563eb', marker='.')
        if val_x:
            loss_axis.plot(val_x, val_y, label='Validation', color='#dc2626', marker='.')
        if train_x or val_x:
            loss_axis.legend(loc='best')
        else:
            loss_axis.text(0.5, 0.5, 'Waiting for loss history…', ha='center', va='center', transform=loss_axis.transAxes)
        loss_axis.grid(True, alpha=0.25)
        lines = [f'Status: {status}', f'Model: {model_label}', f'bestVal: {best_label}']
        run_id = _first(state, 'run_id')
        if run_id:
            lines.append(f'Run: {run_id}')
        if run_path is None:
            lines.extend(['', 'Waiting for a run directory…'])
        else:
            phase = values.get('phase')
            lines.append(f"Phase: {phase or '—'}")
            epoch = values.get('epoch')
            max_epochs = values.get('max_epochs')
            if epoch is not None:
                epoch_text = f'{(int(epoch) if epoch.is_integer() else epoch):g}'
                if max_epochs is not None:
                    max_text = f'{(int(max_epochs) if max_epochs.is_integer() else max_epochs):g}'
                    epoch_text += f' / {max_text}'
                lines.append(f'Epoch: {epoch_text}')
            batch = values.get('batch')
            total_batches = values.get('total_batches')
            if batch is not None:
                batch_text = f'{(int(batch) if batch.is_integer() else batch):g}'
                if total_batches is not None:
                    total_text = f'{(int(total_batches) if total_batches.is_integer() else total_batches):g}'
                    batch_text += f' / {total_text}'
                lines.append(f'Batch: {batch_text}')
            elapsed = _duration(values.get('elapsed'))
            eta = _duration(values.get('eta'))
            if elapsed is not None:
                lines.append(f'Elapsed: {elapsed}')
            if eta is not None:
                lines.append(f'ETA: {eta}')
            current_loss = values.get('loss')
            if current_loss is None:
                current_loss = values.get('train_loss')
            if current_loss is not None:
                lines.append(f'Current loss: {_display_number(current_loss)}')
            if not (run_path / 'progress.json').exists():
                lines.extend(['', 'Waiting for trainer progress…'])
        if cache['notice']:
            lines.extend(['', cache['notice']])
        info_text.set_text('\n'.join(lines))
        cache['can_pause'] = run_path is not None and status.lower() == 'training'
        pause_button.set_active(cache['can_pause'])
        figure.canvas.draw_idle()

    def request_pause(_event: Any) -> None:
        if not cache.get('can_pause'):
            cache['notice'] = 'No active training phase to pause.'
            return
        run_path = cache.get('run_path')
        if run_path is None:
            cache['notice'] = 'No active run is available to pause.'
            return
        stop_file = run_path / 'STOP_REQUESTED'
        try:
            with stop_file.open('x', encoding='utf-8', newline='\n') as stream:
                stream.write(datetime.now().astimezone().isoformat(timespec='seconds') + '\n')
            cache['notice'] = 'Pause requested; waiting for trainer to save and stop.'
        except FileExistsError:
            cache['notice'] = 'A pause request already exists for this run.'
        except OSError as exc:
            cache['notice'] = f'Could not create pause request: {exc}'
        figure.canvas.draw_idle()
    pause_button.on_clicked(request_pause)
    animation = FuncAnimation(figure, refresh, interval=2000, cache_frame_data=False)
    figure._live_view_animation = animation
    refresh(0)
    plt.show(block=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
