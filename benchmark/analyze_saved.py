from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
from typing import Any
METRICS = ('Dice', 'IoU', 'Precision', 'Recall', 'background_Dice', 'background_IoU', 'background_Precision', 'background_Recall', 'HD95_network_px', 'ASSD_network_px', 'BoundaryF1_1px')
BOUNDED_METRICS = {'Dice', 'IoU', 'Precision', 'Recall', 'background_Dice', 'background_IoU', 'background_Precision', 'background_Recall', 'BoundaryF1_1px'}
UNITS = {'Dice': 'ratio', 'IoU': 'ratio', 'Precision': 'ratio', 'Recall': 'ratio', 'background_Dice': 'ratio', 'background_IoU': 'ratio', 'background_Precision': 'ratio', 'background_Recall': 'ratio', 'HD95_network_px': 'network pixels', 'ASSD_network_px': 'network pixels', 'BoundaryF1_1px': 'ratio'}
REQUIRED_COLUMNS = ('sample_id', 'group', 'Dice', 'IoU')
STRATA = (('<=1%', '0 <= foreground_fraction <= 0.01', lambda x: np_less_equal(x, 0.01)), ('>1%-5%', '0.01 < foreground_fraction <= 0.05', lambda x: np_greater(x, 0.01) & np_less_equal(x, 0.05)), ('>5%', 'foreground_fraction > 0.05', lambda x: np_greater(x, 0.05)), ('Unavailable', 'foreground_fraction is blank or NaN', lambda x: ~np_isfinite(x)))

def np_isfinite(values: Any) -> Any:
    import numpy as np
    return np.isfinite(values)

def np_less_equal(values: Any, threshold: float) -> Any:
    import numpy as np
    return np.asarray(values) <= threshold

def np_greater(values: Any, threshold: float) -> Any:
    import numpy as np
    return np.asarray(values) > threshold

def _is_within(path: Path, directory: Path) -> bool:
    candidate = os.path.normcase(str(path.resolve(strict=True)))
    parent = os.path.normcase(str(directory.resolve(strict=True)))
    try:
        return os.path.commonpath((candidate, parent)) == parent
    except ValueError:
        return False

def _resolve_input_file(directory: Path, name: str) -> Path:
    path = directory / name
    if not path.is_file():
        raise ValueError(f'{path}: required input file is missing')
    resolved = path.resolve(strict=True)
    if not _is_within(resolved, directory):
        raise ValueError(f'{path}: input file resolves outside the evaluation directory')
    return resolved

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _validate_inventory(directory: Path) -> dict[str, Any]:
    inventory_path = directory / 'artifact_inventory.json'
    if not inventory_path.exists() and (not inventory_path.is_symlink()):
        return {'present': False, 'validated_file_count': 0, 'sha256': None}
    inventory_path = _resolve_input_file(directory, 'artifact_inventory.json')
    try:
        with inventory_path.open('r', encoding='utf-8-sig') as stream:
            inventory = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'{inventory_path}: could not read valid artifact inventory ({exc})') from exc
    if not isinstance(inventory, dict) or not isinstance(inventory.get('files'), list):
        raise ValueError(f'{inventory_path}: expected an object with a files list')
    root = directory.resolve(strict=True)
    seen: set[str] = set()
    for index, entry in enumerate(inventory['files'], start=1):
        prefix = f'{inventory_path}: files[{index - 1}]'
        if not isinstance(entry, dict):
            raise ValueError(f'{prefix} must be an object')
        raw_relative = entry.get('path')
        if not isinstance(raw_relative, str) or not raw_relative.strip():
            raise ValueError(f'{prefix}.path must be a nonempty relative path')
        if '\x00' in raw_relative:
            raise ValueError(f'{prefix}.path contains a NUL character')
        normalized = raw_relative.replace('\\', '/')
        posix_path = PurePosixPath(normalized)
        windows_path = PureWindowsPath(raw_relative)
        if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
            raise ValueError(f'{prefix}.path must be relative: {raw_relative!r}')
        parts = tuple((part for part in posix_path.parts if part not in ('', '.')))
        if not parts or any((part == '..' for part in parts)):
            raise ValueError(f'{prefix}.path escapes the evaluation directory: {raw_relative!r}')
        normalized_key = '/'.join(parts)
        if normalized_key in seen:
            raise ValueError(f'{prefix}.path is duplicated: {normalized_key!r}')
        seen.add(normalized_key)
        expected_bytes = entry.get('bytes')
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise ValueError(f'{prefix}.bytes must be a nonnegative integer')
        expected_hash = entry.get('sha256')
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f'{prefix}.sha256 must be a 64-character SHA-256 digest')
        try:
            int(expected_hash, 16)
        except ValueError as exc:
            raise ValueError(f'{prefix}.sha256 is not hexadecimal') from exc
        candidate = root.joinpath(*parts)
        if not candidate.is_file():
            raise ValueError(f'{prefix}: listed file is missing: {normalized_key}')
        resolved = candidate.resolve(strict=True)
        if not _is_within(resolved, root):
            raise ValueError(f'{prefix}: listed path resolves outside the evaluation directory: {normalized_key}')
        actual_bytes = resolved.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(f'{prefix}: size mismatch for {normalized_key}: expected {expected_bytes}, got {actual_bytes}')
        actual_hash = _sha256(resolved)
        if actual_hash.lower() != expected_hash.lower():
            raise ValueError(f'{prefix}: SHA-256 mismatch for {normalized_key}')
    return {'present': True, 'validated_file_count': len(inventory['files']), 'sha256': _sha256(inventory_path)}

def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        with path.open('r', encoding='utf-8-sig') as stream:
            metadata = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'{path}: could not read valid evaluation metadata ({exc})') from exc
    if not isinstance(metadata, dict):
        raise ValueError(f'{path}: expected a JSON object')
    for field in ('model', 'dataset_fingerprint', 'protocol'):
        value = metadata.get(field)
        if value is None or value == '' or value == {} or (value == []):
            raise ValueError(f'{path}: missing or empty required metadata field {field!r}')
    if not isinstance(metadata['model'], str) or not metadata['model'].strip():
        raise ValueError(f'{path}: model must be a nonempty string')
    if not isinstance(metadata['dataset_fingerprint'], str) or not metadata['dataset_fingerprint'].strip():
        raise ValueError(f'{path}: dataset_fingerprint must be a nonempty string')
    if not isinstance(metadata['protocol'], (str, dict, list)):
        raise ValueError(f'{path}: protocol must be a nonempty string, object, or list')
    return metadata

def _parse_optional_float(value: str | None, field: str, line: int, source: Path) -> float:
    if value is None or not value.strip():
        return math.nan
    try:
        numeric = float(value)
    except ValueError as exc:
        raise ValueError(f'{source}: row {line}: {field} must be numeric, blank, or NaN') from exc
    if math.isnan(numeric):
        return math.nan
    if not math.isfinite(numeric):
        raise ValueError(f'{source}: row {line}: {field} must be finite or NaN')
    return numeric

def _parse_flag_values(raw: dict[str, str | None], names: tuple[str, ...], line: int, source: Path) -> bool | None:
    values: list[bool] = []
    for name in names:
        if name not in raw or raw[name] is None or (not raw[name].strip()):
            continue
        value = raw[name].strip().casefold()
        if value in {'true', '1', 'yes'}:
            values.append(True)
        elif value in {'false', '0', 'no'}:
            values.append(False)
        else:
            raise ValueError(f'{source}: row {line}: {name} must be a boolean or blank')
    if len(set(values)) > 1:
        raise ValueError(f'{source}: row {line}: conflicting aliases for {names[0]}')
    return values[0] if values else None

def _load_cases(path: Path) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames or []
            if len(fieldnames) != len(set(fieldnames)):
                raise ValueError(f'{path}: duplicate CSV column names')
            missing = [field for field in REQUIRED_COLUMNS if field not in fieldnames]
            if missing:
                raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
            metric_present = {metric: metric in fieldnames for metric in METRICS}
            rows: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            for line, raw in enumerate(reader, start=2):
                sample_id = (raw.get('sample_id') or '').strip()
                group = (raw.get('group') or '').strip()
                if not sample_id:
                    raise ValueError(f'{path}: row {line}: sample_id must be nonempty')
                if not group:
                    raise ValueError(f'{path}: row {line}: group must be nonempty')
                if sample_id in seen_ids:
                    raise ValueError(f'{path}: row {line}: duplicate sample_id {sample_id!r}')
                seen_ids.add(sample_id)
                row: dict[str, Any] = {'sample_id': sample_id, 'group': group, 'empty_prediction': _parse_flag_values(raw, ('empty_prediction', 'prediction_empty'), line, path), 'empty_reference': _parse_flag_values(raw, ('empty_reference', 'reference_empty'), line, path)}
                for metric in METRICS:
                    value = _parse_optional_float(raw.get(metric), metric, line, path) if metric_present[metric] else math.nan
                    if math.isfinite(value):
                        if metric in BOUNDED_METRICS and (not 0.0 <= value <= 1.0):
                            raise ValueError(f'{path}: row {line}: {metric} must be within 0..1')
                        if metric in {'HD95_network_px', 'ASSD_network_px'} and value < 0.0:
                            raise ValueError(f'{path}: row {line}: {metric} must be nonnegative')
                    row[metric] = value
                fraction = _parse_optional_float(raw.get('foreground_fraction'), 'foreground_fraction', line, path)
                if math.isfinite(fraction) and (not 0.0 <= fraction <= 1.0):
                    raise ValueError(f'{path}: row {line}: foreground_fraction must be within 0..1')
                row['foreground_fraction'] = fraction
                rows.append(row)
    except OSError as exc:
        raise ValueError(f'{path}: could not read CSV ({exc})') from exc
    if not rows:
        raise ValueError(f'{path}: no per-case rows found')
    return (rows, metric_present)

def _describe(values: Any, *, n_total: int, field_present: bool) -> dict[str, Any]:
    import numpy as np
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    n_valid = int(finite.size)
    if n_valid:
        mean = float(np.mean(finite))
        median, q1, q3 = (float(x) for x in np.quantile(finite, [0.5, 0.25, 0.75]))
        sample_sd = float(np.std(finite, ddof=1)) if n_valid > 1 else None
    else:
        mean = median = q1 = q3 = sample_sd = None
    return {'field_present': bool(field_present), 'n_total': int(n_total), 'n_valid': n_valid, 'n_invalid': int(n_total - n_valid), 'mean': mean, 'sample_sd': sample_sd, 'median': median, 'q1': q1, 'q3': q3}

def _group_bootstrap_means(values: Any, group_codes: Any, group_count: int, resamples: int, seed: int) -> Any:
    import numpy as np
    values = np.asarray(values, dtype=float)
    group_codes = np.asarray(group_codes, dtype=int)
    if values.ndim != 2:
        raise ValueError('bootstrap values must have shape (cases, metrics)')
    if values.shape[0] != group_codes.size:
        raise ValueError('bootstrap case values and group codes must have equal length')
    if group_count <= 0 or resamples <= 0:
        raise ValueError('bootstrap requires at least one group and one resample')
    if group_codes.size == 0 or np.any(group_codes < 0) or np.any(group_codes >= group_count):
        raise ValueError('bootstrap group codes are empty or outside the declared group range')
    n_metrics = values.shape[1]
    finite = np.isfinite(values)
    group_sums = np.zeros((group_count, n_metrics), dtype=float)
    group_finite_counts = np.zeros((group_count, n_metrics), dtype=float)
    np.add.at(group_sums, group_codes, np.where(finite, values, 0.0))
    np.add.at(group_finite_counts, group_codes, finite.astype(float))
    rng = np.random.default_rng(seed)
    means = np.full((resamples, n_metrics), np.nan, dtype=float)
    chunk_size = max(1, min(resamples, 2000000 // max(group_count, 1)))
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        batch = stop - start
        draws = rng.integers(0, group_count, size=(batch, group_count))
        weights = np.zeros((batch, group_count), dtype=float)
        sample_rows = np.repeat(np.arange(batch), group_count)
        np.add.at(weights, (sample_rows, draws.reshape(-1)), 1.0)
        numerators = weights @ group_sums
        denominators = weights @ group_finite_counts
        block = means[start:stop]
        np.divide(numerators, denominators, out=block, where=denominators > 0.0)
    return means

def _percentile_ci(values: Any) -> tuple[float | None, float | None, int]:
    import numpy as np
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (None, None, 0)
    low, high = np.quantile(finite, [0.025, 0.975])
    return (float(low), float(high), int(finite.size))

def _analysis_rows(rows: list[dict[str, Any]], metric_present: dict[str, bool], resamples: int, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    import numpy as np
    n_cases = len(rows)
    group_labels = sorted({row['group'] for row in rows})
    group_index = {label: index for index, label in enumerate(group_labels)}
    group_codes = np.asarray([group_index[row['group']] for row in rows], dtype=int)
    values = np.asarray([[row[metric] for metric in METRICS] for row in rows], dtype=float)
    bootstrap = _group_bootstrap_means(values, group_codes, len(group_labels), resamples, seed)
    metric_objects: dict[str, Any] = {}
    metric_csv: list[dict[str, Any]] = []
    for position, metric in enumerate(METRICS):
        description = _describe(values[:, position], n_total=n_cases if metric_present[metric] else 0, field_present=metric_present[metric])
        ci_low, ci_high, valid_replicates = _percentile_ci(bootstrap[:, position])
        description.update({'unit': UNITS[metric], 'group_bootstrap_ci95_low': ci_low, 'group_bootstrap_ci95_high': ci_high, 'bootstrap_valid_replicates': valid_replicates, 'bootstrap_invalid_replicates': int(resamples - valid_replicates)})
        metric_objects[metric] = description
        metric_csv.append({'metric': metric, **description})
    fractions = np.asarray([row['foreground_fraction'] for row in rows], dtype=float)
    dice = values[:, METRICS.index('Dice')]
    boundary = values[:, METRICS.index('BoundaryF1_1px')]
    dice_valid = np.isfinite(dice)
    boundary_valid = np.isfinite(boundary)
    empty_prediction_known = np.asarray([row['empty_prediction'] is not None for row in rows])
    empty_prediction_true = np.asarray([row['empty_prediction'] is True for row in rows])
    empty_reference_known = np.asarray([row['empty_reference'] is not None for row in rows])
    empty_reference_true = np.asarray([row['empty_reference'] is True for row in rows])
    thresholds = {'Dice_below_0_5': {'count': int(np.count_nonzero(dice_valid & (dice < 0.5))), 'valid_dice_n': int(np.count_nonzero(dice_valid)), 'invalid_dice_n': int(n_cases - np.count_nonzero(dice_valid)), 'threshold': 'Dice < 0.5'}, 'Dice_below_0_7': {'count': int(np.count_nonzero(dice_valid & (dice < 0.7))), 'valid_dice_n': int(np.count_nonzero(dice_valid)), 'invalid_dice_n': int(n_cases - np.count_nonzero(dice_valid)), 'threshold': 'Dice < 0.7'}, 'boundary_failures': {'count': int(np.count_nonzero(boundary_valid & (boundary == 0.0))), 'valid_boundary_f1_n': int(np.count_nonzero(boundary_valid)), 'invalid_boundary_f1_n': int(n_cases - np.count_nonzero(boundary_valid)), 'definition': 'BoundaryF1_1px == 0 among finite values; no boundary matches within 1 network pixel.'}}
    stratum_arrays: list[tuple[str, str, Any]] = []
    for label, rule, classifier in STRATA:
        mask = classifier(fractions)
        stratum_arrays.append((label, rule, mask))
    stratified_csv: list[dict[str, Any]] = []
    strata_summary: list[dict[str, Any]] = []
    for label, rule, mask in stratum_arrays:
        stratum_n = int(np.count_nonzero(mask))
        stratum_dice = dice[mask]
        stratum_boundary = boundary[mask]
        stratum_dice_valid = np.isfinite(stratum_dice)
        stratum_boundary_valid = np.isfinite(stratum_boundary)
        stratum_counts = {'dice_below_0_5_n': int(np.count_nonzero(stratum_dice_valid & (stratum_dice < 0.5))), 'dice_valid_n': int(np.count_nonzero(stratum_dice_valid)), 'dice_below_0_7_n': int(np.count_nonzero(stratum_dice_valid & (stratum_dice < 0.7))), 'boundary_failures_n': int(np.count_nonzero(stratum_boundary_valid & (stratum_boundary == 0.0))), 'boundary_f1_valid_n': int(np.count_nonzero(stratum_boundary_valid))}
        strata_summary.append({'stratum': label, 'foreground_fraction_rule': rule, 'case_count': stratum_n, 'foreground_fraction_valid_n': int(np.count_nonzero(mask & np.isfinite(fractions))), **stratum_counts})
        for position, metric in enumerate(METRICS):
            stats = _describe(values[mask, position], n_total=stratum_n if metric_present[metric] else 0, field_present=metric_present[metric])
            stratified_csv.append({'stratum': label, 'foreground_fraction_rule': rule, 'case_count': stratum_n, 'foreground_fraction_valid_n': int(np.count_nonzero(mask & np.isfinite(fractions))), 'metric': metric, **stats, 'unit': UNITS[metric], **stratum_counts})
    summary = {'analysis': 'offline_single_model_saved_evaluation', 'case_count': n_cases, 'group_count': len(group_labels), 'group_labels': group_labels, 'bootstrap': {'method': '95% percentile cluster bootstrap', 'sampling_unit': 'group, sampled with replacement', 'resamples': resamples, 'seed': seed, 'mean_unit': 'image-macro; every case in each drawn group is retained, with finite-case denominators per metric', 'group_count': len(group_labels), 'valid_replicates_reported_per_metric': True}, 'metrics': metric_objects, 'threshold_counts': thresholds, 'empty_case_flags': {'empty_prediction_true_n': int(np.count_nonzero(empty_prediction_true)), 'empty_prediction_flag_valid_n': int(np.count_nonzero(empty_prediction_known)), 'empty_prediction_flag_missing_n': int(n_cases - np.count_nonzero(empty_prediction_known)), 'empty_prediction_with_finite_dice_n': int(np.count_nonzero(empty_prediction_true & dice_valid)), 'empty_reference_true_n': int(np.count_nonzero(empty_reference_true)), 'empty_reference_flag_valid_n': int(np.count_nonzero(empty_reference_known)), 'empty_reference_flag_missing_n': int(n_cases - np.count_nonzero(empty_reference_known)), 'empty_prediction_dice_handling': 'No filtering by empty_prediction; all finite saved Dice values, including finite zeros, are included.'}, 'stratification': {'variable': 'foreground_fraction from the reference mask', 'labels_are_descriptive': 'These bins describe reference foreground area only; they do not define clinical rarity, severity, or acceptability.', 'bins': strata_summary}, 'uncertainty_caveat': 'Intervals describe group-resampling uncertainty for this one saved model and act as group-proxy limits. They do not measure variation across training seeds, retraining, or independent model fits.', 'boundary_units': {'HD95_network_px': 'network pixels', 'ASSD_network_px': 'network pixels', 'physical_calibration': None, 'statement': 'No physical spacing or calibration was used; HD95 and ASSD are reported in network pixels.'}}
    return (summary, metric_csv, stratified_csv)

def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def _plot_distributions(out: Path, rows: list[dict[str, Any]], seed: int) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    rng = np.random.default_rng(seed)

    def draw_panel(ax: Any, metrics: tuple[str, ...], title: str, bounded: bool=False) -> None:
        for position, metric in enumerate(metrics, start=1):
            values = np.asarray([row[metric] for row in rows], dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                box = ax.boxplot([values], positions=[position], widths=0.48, showfliers=False, patch_artist=True)
                for patch in box['boxes']:
                    patch.set_facecolor('#b8cde8')
                    patch.set_edgecolor('#466789')
                jitter = rng.uniform(-0.12, 0.12, size=values.size)
                ax.scatter(position + jitter, values, s=13, alpha=0.62, color='#244b70', linewidths=0, zorder=3)
            ax.text(position, 0.02 if bounded else 0.98, f'n={values.size}', ha='center', va='bottom' if bounded else 'top', fontsize=8, transform=ax.get_xaxis_transform())
        ax.set_xticks(range(1, len(metrics) + 1))
        ax.set_xticklabels(metrics, rotation=14, ha='right')
        ax.set_title(title)
        ax.grid(axis='y', color='#dddddd', linewidth=0.7)
        ax.set_axisbelow(True)
        if bounded:
            ax.set_ylim(-0.03, 1.03)
            ax.set_ylabel('Image-level score')
        else:
            ax.set_ylabel('Image-level score')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    draw_panel(axes[0], ('Dice', 'IoU'), 'Foreground overlap', bounded=True)
    draw_panel(axes[1], ('background_Dice', 'background_IoU'), 'Background overlap', bounded=True)
    fig.suptitle('Saved per-image metrics; dots are cases, boxes show median and interquartile range')
    fig.tight_layout()
    fig.savefig(out / 'distributions.png', dpi=160)
    plt.close(fig)
    boundary_specs = (('HD95_network_px', 'HD95', 'Network pixels', False), ('ASSD_network_px', 'ASSD', 'Network pixels', False), ('BoundaryF1_1px', 'Boundary F1 at 1 px', 'Ratio', True))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (metric, label, unit, bounded) in zip(axes, boundary_specs):
        values = np.asarray([row[metric] for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            box = ax.boxplot([values], positions=[1], widths=0.5, showfliers=False, patch_artist=True)
            for patch in box['boxes']:
                patch.set_facecolor('#c3ddcb')
                patch.set_edgecolor('#467558')
            jitter = rng.uniform(-0.12, 0.12, size=values.size)
            ax.scatter(1 + jitter, values, s=13, alpha=0.62, color='#2b6740', linewidths=0, zorder=3)
        ax.set_xticks([1])
        ax.set_xticklabels([f'finite pairs n={values.size}'])
        ax.set_title(label)
        ax.set_ylabel(unit)
        ax.grid(axis='y', color='#dddddd', linewidth=0.7)
        ax.set_axisbelow(True)
        if bounded:
            ax.set_ylim(-0.03, 1.03)
    fig.suptitle('Boundary metrics; HD95 and ASSD use network pixels (no physical calibration)')
    fig.tight_layout()
    fig.savefig(out / 'boundary_distributions.png', dpi=160)
    plt.close(fig)

def _resolve_output(output: str | Path, evaluation: Path) -> Path:
    requested = Path(output).expanduser()
    if requested.exists() or requested.is_symlink():
        raise ValueError(f'--out must be a new directory and must not already exist: {requested}')
    resolved = requested.resolve(strict=False)
    evaluation_resolved = evaluation.resolve(strict=True)
    output_key = os.path.normcase(str(resolved))
    evaluation_key = os.path.normcase(str(evaluation_resolved))
    try:
        contained = os.path.commonpath((output_key, evaluation_key)) == evaluation_key
    except ValueError:
        contained = False
    if contained:
        raise ValueError(f'--out must not be inside the evaluation directory: {resolved}')
    return resolved

def _write_outputs(out: Path, summary: dict[str, Any], metric_rows: list[dict[str, Any]], stratified_rows: list[dict[str, Any]], source_metadata: dict[str, Any], source_hashes: dict[str, str], inventory_status: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=False)
    summary.update({'source_metadata': source_metadata, 'source_hashes_sha256': source_hashes, 'artifact_inventory_validation': inventory_status})
    with (out / 'summary.json').open('w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    metric_fields = ['metric', 'unit', 'field_present', 'n_total', 'n_valid', 'n_invalid', 'mean', 'sample_sd', 'median', 'q1', 'q3', 'group_bootstrap_ci95_low', 'group_bootstrap_ci95_high', 'bootstrap_valid_replicates', 'bootstrap_invalid_replicates']
    _write_csv(out / 'metric_summary.csv', metric_rows, metric_fields)
    stratified_fields = ['stratum', 'foreground_fraction_rule', 'case_count', 'foreground_fraction_valid_n', 'metric', 'unit', 'field_present', 'n_total', 'n_valid', 'n_invalid', 'mean', 'sample_sd', 'median', 'q1', 'q3', 'dice_below_0_5_n', 'dice_valid_n', 'dice_below_0_7_n', 'boundary_failures_n', 'boundary_f1_valid_n']
    _write_csv(out / 'stratified_summary.csv', stratified_rows, stratified_fields)

class _ArgumentParser(argparse.ArgumentParser):

    def error(self, message: str) -> None:
        raise ValueError(message)

def main(argv: list[str] | None=None) -> int:
    parser = _ArgumentParser(description='Analyze saved per-case segmentation metrics without retraining or inference.')
    parser.add_argument('--evaluation', required=True, help='saved evaluation directory')
    parser.add_argument('--out', required=True, help='new output directory (must not already exist)')
    parser.add_argument('--resamples', type=int, default=10000, help='group-bootstrap replicates (default: 10000)')
    parser.add_argument('--seed', type=int, default=20261105, help='bootstrap and plot seed (default: 20261105)')
    try:
        args = parser.parse_args(argv)
        if args.resamples <= 0:
            raise ValueError('--resamples must be a positive integer')
        if args.seed < 0:
            raise ValueError('--seed must be a nonnegative integer')
        evaluation = Path(args.evaluation).expanduser().resolve(strict=True)
        if not evaluation.is_dir():
            raise ValueError(f'--evaluation is not a directory: {evaluation}')
        out = _resolve_output(args.out, evaluation)
        metadata_path = _resolve_input_file(evaluation, 'evaluation.json')
        metrics_path = _resolve_input_file(evaluation, 'per_case_metrics.csv')
        inventory_status = _validate_inventory(evaluation)
        metadata = _read_metadata(metadata_path)
        rows, metric_present = _load_cases(metrics_path)
        summary, metric_rows, stratified_rows = _analysis_rows(rows, metric_present, args.resamples, args.seed)
        summary['model'] = metadata['model']
        summary['dataset_fingerprint'] = metadata['dataset_fingerprint']
        summary['protocol'] = metadata['protocol']
        source_hashes = {'evaluation.json': _sha256(metadata_path), 'per_case_metrics.csv': _sha256(metrics_path)}
        if inventory_status['present']:
            source_hashes['artifact_inventory.json'] = inventory_status['sha256']
        import matplotlib
        import numpy
        _write_outputs(out, summary, metric_rows, stratified_rows, metadata, source_hashes, inventory_status)
        _plot_distributions(out, rows, args.seed)
        print(f'Offline analysis written to {out}')
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f'analyze_saved: {exc}', file=sys.stderr)
        return 2
if __name__ == '__main__':
    raise SystemExit(main())
