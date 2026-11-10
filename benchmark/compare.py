from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
METRICS = ('Dice', 'IoU')
CSV_FIELDS = ('sample_id', 'group', 'tp', 'fp', 'fn', 'tn', 'Dice', 'IoU', 'Precision', 'Recall')
COUNT_FIELDS = ('tp', 'fp', 'fn', 'tn')
CI_LEVEL = 0.95
VALUE_TOLERANCE = 1e-06
TIE_TOLERANCE = 1e-12

@dataclass
class Run:
    path: Path
    metadata: dict[str, Any]
    rows: list[dict[str, Any]]
    by_id: dict[str, dict[str, Any]]
    source_hashes: dict[str, str]

    @property
    def model(self) -> str:
        return str(self.metadata['model']).strip()

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'item'):
        return value.item()
    return value

def _parse_count(value: str, field: str, line: int, source: Path) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{source}: row {line}: {field} must be a nonnegative integer') from exc
    if not math.isfinite(numeric) or numeric < 0 or (not numeric.is_integer()):
        raise ValueError(f'{source}: row {line}: {field} must be a nonnegative integer')
    return int(numeric)

def _parse_metric(value: str, field: str, line: int, source: Path) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{source}: row {line}: {field} must be numeric') from exc
    if not math.isfinite(numeric):
        raise ValueError(f'{source}: row {line}: {field} must be finite')
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f'{source}: row {line}: {field} must be within 0..1')
    return numeric

def _parse_optional_metric(value: str | None, field: str, line: int, source: Path) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        numeric = float(value)
    except ValueError as exc:
        raise ValueError(f'{source}: row {line}: {field} must be numeric or blank') from exc
    if math.isnan(numeric):
        return None
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f'{source}: row {line}: {field} must be within 0..1 when defined')
    return numeric

def _validate_count_metrics(row: dict[str, Any], line: int, source: Path) -> None:
    dice_denominator = 2 * row['tp'] + row['fp'] + row['fn']
    iou_denominator = row['tp'] + row['fp'] + row['fn']
    if dice_denominator == 0 or iou_denominator == 0:
        raise ValueError(f'{source}: row {line}: empty-empty overlap is undefined under this protocol; define and report a missing-case analysis before comparing such a cohort')
    expected_dice = 2.0 * row['tp'] / dice_denominator
    expected_iou = row['tp'] / iou_denominator
    for metric, expected in (('Dice', expected_dice), ('IoU', expected_iou)):
        if not math.isclose(row[metric], expected, rel_tol=VALUE_TOLERANCE, abs_tol=VALUE_TOLERANCE):
            raise ValueError(f'{source}: row {line}: {metric}={row[metric]:.12g} is inconsistent with counts (expected {expected:.12g})')

def _parse_seed(value: Any, source: Path) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f'{source}: metadata seed must be an integer')
    try:
        seed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{source}: metadata seed must be an integer') from exc
    if isinstance(value, float) and (not value.is_integer()):
        raise ValueError(f'{source}: metadata seed must be an integer')
    if isinstance(value, str) and str(seed) != value.strip():
        raise ValueError(f'{source}: metadata seed must be an integer')
    return seed

def load_run(directory: str | Path) -> Run:
    path = Path(directory).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f'evaluation path is not a directory: {path}')
    metadata_path = path / 'evaluation.json'
    metrics_path = path / 'per_case_metrics.csv'
    if not metadata_path.is_file() or not metrics_path.is_file():
        raise ValueError(f'{path}: expected evaluation.json and per_case_metrics.csv')
    try:
        with metadata_path.open('r', encoding='utf-8-sig') as source:
            metadata = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'{metadata_path}: could not read valid JSON ({exc})') from exc
    if not isinstance(metadata, dict):
        raise ValueError(f'{metadata_path}: expected a JSON object')
    for field in ('dataset_fingerprint', 'protocol', 'checkpoint_sha256', 'model', 'split', 'seed'):
        if field not in metadata:
            raise ValueError(f'{metadata_path}: missing required metadata field {field!r}')
    if metadata['split'] != 'test':
        raise ValueError(f"{metadata_path}: split must be 'test', got {metadata['split']!r}")
    for field in ('dataset_fingerprint', 'checkpoint_sha256', 'model'):
        if not isinstance(metadata[field], str) or not metadata[field].strip():
            raise ValueError(f'{metadata_path}: {field} must be a nonempty string')
    if not isinstance(metadata['protocol'], (str, dict, list)) or not metadata['protocol']:
        raise ValueError(f'{metadata_path}: protocol must be a nonempty string, object, or list')
    metadata = dict(metadata)
    metadata['model'] = metadata['model'].strip()
    metadata['seed'] = _parse_seed(metadata['seed'], metadata_path)
    rows: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    try:
        with metrics_path.open('r', encoding='utf-8-sig', newline='') as source:
            reader = csv.DictReader(source)
            header = set(reader.fieldnames or ())
            missing = [field for field in CSV_FIELDS if field not in header]
            if missing:
                raise ValueError(f"{metrics_path}: missing CSV fields: {', '.join(missing)}")
            for line, raw in enumerate(reader, start=2):
                sample_id = (raw.get('sample_id') or '').strip()
                group = (raw.get('group') or '').strip()
                if not sample_id:
                    raise ValueError(f'{metrics_path}: row {line}: sample_id must be nonempty')
                if not group:
                    raise ValueError(f'{metrics_path}: row {line}: group must be nonempty')
                if sample_id in by_id:
                    raise ValueError(f'{metrics_path}: duplicate sample_id {sample_id!r}')
                row: dict[str, Any] = {'sample_id': sample_id, 'group': group}
                for field in COUNT_FIELDS:
                    row[field] = _parse_count(raw.get(field), field, line, metrics_path)
                for field in METRICS:
                    row[field] = _parse_metric(raw.get(field), field, line, metrics_path)
                row['Precision'] = _parse_optional_metric(raw.get('Precision'), 'Precision', line, metrics_path)
                row['Recall'] = _parse_optional_metric(raw.get('Recall'), 'Recall', line, metrics_path)
                _validate_count_metrics(row, line, metrics_path)
                rows.append(row)
                by_id[sample_id] = row
    except OSError as exc:
        raise ValueError(f'{metrics_path}: could not read CSV ({exc})') from exc
    if not rows:
        raise ValueError(f'{metrics_path}: no per-case rows found')
    return Run(path=path, metadata=metadata, rows=rows, by_id=by_id, source_hashes={'evaluation.json': _sha256(metadata_path), 'per_case_metrics.csv': _sha256(metrics_path)})

def safe_slug(label: str) -> str:
    slug = re.sub('[^A-Za-z0-9]+', '_', label.strip()).strip('_').lower()
    return slug or 'model'

def validate_runs(runs: list[Run]) -> tuple[list[str], list[str]]:
    labels: dict[str, str] = {}
    slugs: dict[str, str] = {}
    paths: set[Path] = set()
    for run in runs:
        if run.path in paths:
            raise ValueError(f'evaluation directory supplied more than once: {run.path}')
        paths.add(run.path)
        label_key = ' '.join(run.model.casefold().split())
        if label_key in labels:
            raise ValueError(f'duplicate model label: {run.model!r} and {labels[label_key]!r}')
        labels[label_key] = run.model
        slug = safe_slug(run.model)
        if slug in slugs:
            raise ValueError(f'output-name collision between model labels {run.model!r} and {slugs[slug]!r}')
        slugs[slug] = run.model
    reference = runs[0]
    reference_fingerprint = reference.metadata['dataset_fingerprint']
    protocol_key = json.dumps(reference.metadata['protocol'], sort_keys=True, separators=(',', ':'))
    reference_ids = set(reference.by_id)
    reference_groups = {sample_id: row['group'] for sample_id, row in reference.by_id.items()}
    for run in runs[1:]:
        if run.metadata['dataset_fingerprint'] != reference_fingerprint:
            raise ValueError(f'dataset_fingerprint mismatch: {reference.model!r} vs {run.model!r}')
        run_protocol = json.dumps(run.metadata['protocol'], sort_keys=True, separators=(',', ':'))
        if run_protocol != protocol_key:
            raise ValueError(f'protocol mismatch: {reference.model!r} vs {run.model!r}')
        run_ids = set(run.by_id)
        if run_ids != reference_ids:
            missing = sorted(reference_ids - run_ids)[:5]
            extra = sorted(run_ids - reference_ids)[:5]
            raise ValueError(f'case ID mismatch for {run.model!r}; missing examples={missing}, extra examples={extra}')
        mismatched = [sample_id for sample_id in reference_ids if run.by_id[sample_id]['group'] != reference_groups[sample_id]]
        if mismatched:
            raise ValueError(f'group mapping mismatch for {run.model!r}; sample examples={sorted(mismatched)[:5]}')
        for sample_id in reference_ids:
            a, b = (reference.by_id[sample_id], run.by_id[sample_id])
            if a['tp'] + a['fn'] != b['tp'] + b['fn'] or sum((a[k] for k in COUNT_FIELDS)) != sum((b[k] for k in COUNT_FIELDS)):
                raise ValueError(f'Ground-truth foreground/valid-pixel mismatch: {run.model}, {sample_id}')
    ids = sorted(reference_ids)
    groups = sorted({reference.by_id[sample_id]['group'] for sample_id in ids})
    return (ids, groups)

def resolve_output_path(output: str | Path, runs: list[Run]) -> Path:
    path = Path(output).expanduser().resolve()
    for run in runs:
        try:
            path.relative_to(run.path)
        except ValueError:
            pass
        else:
            raise ValueError(f'--out must not be inside an input evaluation directory: {path}')
    if path.exists():
        if not path.is_dir():
            raise ValueError(f'--out exists and is not a directory: {path}')
        try:
            next(path.iterdir())
        except StopIteration:
            pass
        else:
            raise ValueError(f'--out must be a new or empty directory; output collision at {path}')
    return path

def _bootstrap(values: Any, group_index: Any, n_groups: int, resamples: int, seed: int) -> tuple[Any, Any]:
    import numpy as np
    n_cases, n_models, n_metrics = values.shape
    group_sizes = np.bincount(group_index, minlength=n_groups).astype(float)
    group_sums = np.zeros((n_groups, n_models, n_metrics), dtype=float)
    np.add.at(group_sums, group_index, values)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_groups, size=(resamples, n_groups))
    weights = np.zeros((resamples, n_groups), dtype=float)
    sample_rows = np.repeat(np.arange(resamples), n_groups)
    np.add.at(weights, (sample_rows, draws.reshape(-1)), 1.0)
    denominators = weights @ group_sizes
    numerators = (weights @ group_sums.reshape(n_groups, n_models * n_metrics)).reshape(resamples, n_models, n_metrics)
    means = numerators / denominators[:, None, None]
    differences = means[:, :1, :] - means[:, 1:, :]
    return (means, differences)

def _ci(array: Any) -> tuple[float, float]:
    import numpy as np
    low, high = np.quantile(array, [(1.0 - CI_LEVEL) / 2.0, 1.0 - (1.0 - CI_LEVEL) / 2.0])
    return (float(low), float(high))

def _holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted

def _group_wilcoxon(group_differences: Any) -> tuple[float, float]:
    import numpy as np
    from scipy.stats import wilcoxon
    values = np.asarray(group_differences, dtype=float)
    if np.all(np.abs(values) <= TIE_TOLERANCE):
        return (0.0, 1.0)
    result = wilcoxon(values, zero_method='pratt', alternative='two-sided', method='auto')
    return (float(result.statistic), float(result.pvalue))

def _make_analysis(runs: list[Run], case_ids: list[str], group_labels: list[str], args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    n_cases = len(case_ids)
    n_models = len(runs)
    values = np.empty((n_cases, n_models, len(METRICS)), dtype=float)
    group_index_by_label = {label: index for index, label in enumerate(group_labels)}
    group_index = np.empty(n_cases, dtype=int)
    for case_position, sample_id in enumerate(case_ids):
        group_index[case_position] = group_index_by_label[runs[0].by_id[sample_id]['group']]
        for model_position, run in enumerate(runs):
            values[case_position, model_position, :] = [run.by_id[sample_id][metric] for metric in METRICS]
    means, bootstrap_differences = _bootstrap(values, group_index, len(group_labels), args.resamples, args.seed)
    point_means = values.mean(axis=0)
    model_summaries: list[dict[str, Any]] = []
    for model_position, run in enumerate(runs):
        model_summaries.append({'model': run.model, 'role': 'reference' if model_position == 0 else 'baseline', 'metadata': run.metadata, 'source_sha256': run.source_hashes, 'image_macro_mean': {metric: float(point_means[model_position, metric_position]) for metric_position, metric in enumerate(METRICS)}, 'group_bootstrap_ci95': {metric: list(_ci(means[:, model_position, metric_position])) for metric_position, metric in enumerate(METRICS)}})
    comparison_summaries: list[dict[str, Any]] = []
    wilcoxon_requests: list[tuple[int, str, float, float]] = []
    for baseline_position, baseline in enumerate(runs[1:], start=1):
        metric_results: dict[str, Any] = {}
        for metric_position, metric in enumerate(METRICS):
            case_differences = values[:, 0, metric_position] - values[:, baseline_position, metric_position]
            wins = int(np.count_nonzero(case_differences > TIE_TOLERANCE))
            losses = int(np.count_nonzero(case_differences < -TIE_TOLERANCE))
            ties = n_cases - wins - losses
            group_sums = np.bincount(group_index, weights=case_differences, minlength=len(group_labels))
            group_sizes = np.bincount(group_index, minlength=len(group_labels))
            group_means = group_sums / group_sizes
            p_entry = None
            if args.wilcoxon:
                statistic, p_value = _group_wilcoxon(group_means)
                wilcoxon_requests.append((baseline_position, metric, statistic, p_value))
                p_entry = {'statistic': statistic, 'p_raw': p_value, 'p_holm': None}
            metric_results[metric] = {'reference_minus_baseline_image_macro_mean': float(point_means[0, metric_position] - point_means[baseline_position, metric_position]), 'group_bootstrap_ci95': list(_ci(bootstrap_differences[:, baseline_position - 1, metric_position])), 'case_wins_ties_losses': {'reference_wins': wins, 'ties': ties, 'baseline_wins': losses}, 'group_mean_signed_rank': p_entry}
        comparison_summaries.append({'reference_model': runs[0].model, 'baseline_model': baseline.model, 'metrics': metric_results})
    if args.wilcoxon and wilcoxon_requests:
        adjusted = _holm_adjust([entry[3] for entry in wilcoxon_requests])
        comparison_by_model = {item['baseline_model']: item for item in comparison_summaries}
        for (baseline_position, metric, _statistic, _p_value), p_adjusted in zip(wilcoxon_requests, adjusted):
            target = comparison_by_model[runs[baseline_position].model]['metrics'][metric]['group_mean_signed_rank']
            target['p_holm'] = p_adjusted
    return {'reference_model': runs[0].model, 'settings': {'resamples': args.resamples, 'bootstrap_seed': args.seed, 'confidence_interval': '95% percentile', 'sampling_unit': 'group, sampled with replacement', 'mean_unit': 'image-macro; each sampled group contributes all its cases', 'case_count': n_cases, 'group_count': len(group_labels), 'group_labels': group_labels}, 'models': model_summaries, 'comparisons': comparison_summaries, 'statistical_caveat': 'Bootstrap intervals resample groups and retain all images in each sampled group. Optional Wilcoxon results use group-level mean paired differences as a proxy for independent clusters; the signed-rank test assumes symmetry of those differences. They are descriptive, do not establish independent image-level observations, and must not be interpreted as independent-image p-values. Holm adjustment spans all baseline-by-metric tests in this run.'}

def _plot_pair(out_dir: Path, reference: Run, baseline: Run, case_ids: list[str], comparison: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    slug = safe_slug(baseline.model)
    pair_dir = out_dir / f'pair_{slug}'
    pair_dir.mkdir(parents=False, exist_ok=True)
    reference_values = np.array([[reference.by_id[sample_id][m] for m in METRICS] for sample_id in case_ids])
    baseline_values = np.array([[baseline.by_id[sample_id][m] for m in METRICS] for sample_id in case_ids])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for metric_position, (metric, axis) in enumerate(zip(METRICS, axes)):
        x = baseline_values[:, metric_position]
        y = reference_values[:, metric_position]
        axis.scatter(x, y, s=20, alpha=0.65, edgecolors='none')
        axis.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
        ci_low, ci_high = comparison['metrics'][metric]['group_bootstrap_ci95']
        mean_difference = comparison['metrics'][metric]['reference_minus_baseline_image_macro_mean']
        axis.text(0.04, 0.96, f'mean Δ={mean_difference:+.4f}\n95% group-bootstrap CI [{ci_low:+.4f}, {ci_high:+.4f}]', transform=axis.transAxes, va='top', fontsize=9)
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel=f'{baseline.model} {metric}', ylabel=f'{reference.model} {metric}', title=metric)
        axis.grid(alpha=0.2)
    fig.suptitle(f'Per-case scores: {reference.model} vs {baseline.model}')
    fig.savefig(pair_dir / 'score_scatter.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for metric_position, (metric, axis) in enumerate(zip(METRICS, axes)):
        x = (baseline_values[:, metric_position] + reference_values[:, metric_position]) / 2.0
        differences = reference_values[:, metric_position] - baseline_values[:, metric_position]
        mean_difference = float(differences.mean())
        sd_difference = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
        loa_low = mean_difference - 1.96 * sd_difference
        loa_high = mean_difference + 1.96 * sd_difference
        ci_low, ci_high = comparison['metrics'][metric]['group_bootstrap_ci95']
        axis.scatter(x, differences, s=20, alpha=0.65, edgecolors='none')
        axis.axhline(0, color='gray', linewidth=1)
        axis.axhline(mean_difference, color='tab:blue', linewidth=1.4, label=f'mean Δ={mean_difference:+.4f}')
        axis.axhline(loa_low, color='tab:orange', linestyle='--', linewidth=1, label='descriptive LoA (not CI)')
        axis.axhline(loa_high, color='tab:orange', linestyle='--', linewidth=1)
        axis.text(0.04, 0.96, f'95% group-bootstrap CI [{ci_low:+.4f}, {ci_high:+.4f}]\nLoA are descriptive, not confidence limits', transform=axis.transAxes, va='top', fontsize=9)
        axis.set(xlabel=f'Mean of paired {metric} scores', ylabel=f'{reference.model} − {baseline.model}', title=metric)
        axis.grid(alpha=0.2)
        axis.legend(loc='best', fontsize=8)
    fig.suptitle(f'Bland–Altman style paired differences: {reference.model} vs {baseline.model}')
    fig.savefig(pair_dir / 'bland_altman_style.png', dpi=160)
    plt.close(fig)

def _write_ranked_cases(path: Path, reference: Run, baseline: Run, case_ids: list[str]) -> None:
    fields = ['rank_by_Dice_difference', 'sample_id', 'group', 'reference_Dice', 'baseline_Dice', 'Dice_difference_reference_minus_baseline', 'reference_IoU', 'baseline_IoU', 'IoU_difference_reference_minus_baseline', 'reference_Precision', 'baseline_Precision', 'reference_Recall', 'baseline_Recall', 'Dice_outcome', 'IoU_outcome', 'Dice_reference_wins_total', 'Dice_ties_total', 'Dice_baseline_wins_total', 'IoU_reference_wins_total', 'IoU_ties_total', 'IoU_baseline_wins_total']
    rows: list[dict[str, Any]] = []
    all_dice_differences = [reference.by_id[sample_id]['Dice'] - baseline.by_id[sample_id]['Dice'] for sample_id in case_ids]
    all_iou_differences = [reference.by_id[sample_id]['IoU'] - baseline.by_id[sample_id]['IoU'] for sample_id in case_ids]
    dice_win_counts = (sum((value > TIE_TOLERANCE for value in all_dice_differences)), sum((abs(value) <= TIE_TOLERANCE for value in all_dice_differences)), sum((value < -TIE_TOLERANCE for value in all_dice_differences)))
    iou_win_counts = (sum((value > TIE_TOLERANCE for value in all_iou_differences)), sum((abs(value) <= TIE_TOLERANCE for value in all_iou_differences)), sum((value < -TIE_TOLERANCE for value in all_iou_differences)))
    for sample_id in case_ids:
        ref = reference.by_id[sample_id]
        base = baseline.by_id[sample_id]
        dice_diff = ref['Dice'] - base['Dice']
        iou_diff = ref['IoU'] - base['IoU']
        rows.append({'sample_id': sample_id, 'group': ref['group'], 'reference_Dice': ref['Dice'], 'baseline_Dice': base['Dice'], 'Dice_difference_reference_minus_baseline': dice_diff, 'reference_IoU': ref['IoU'], 'baseline_IoU': base['IoU'], 'IoU_difference_reference_minus_baseline': iou_diff, 'reference_Precision': ref['Precision'], 'baseline_Precision': base['Precision'], 'reference_Recall': ref['Recall'], 'baseline_Recall': base['Recall'], 'Dice_outcome': 'reference_win' if dice_diff > TIE_TOLERANCE else 'baseline_win' if dice_diff < -TIE_TOLERANCE else 'tie', 'IoU_outcome': 'reference_win' if iou_diff > TIE_TOLERANCE else 'baseline_win' if iou_diff < -TIE_TOLERANCE else 'tie', 'Dice_reference_wins_total': dice_win_counts[0], 'Dice_ties_total': dice_win_counts[1], 'Dice_baseline_wins_total': dice_win_counts[2], 'IoU_reference_wins_total': iou_win_counts[0], 'IoU_ties_total': iou_win_counts[1], 'IoU_baseline_wins_total': iou_win_counts[2]})
    rows.sort(key=lambda row: (-row['Dice_difference_reference_minus_baseline'], -row['IoU_difference_reference_minus_baseline'], row['sample_id']))
    for rank, row in enumerate(rows, start=1):
        row['rank_by_Dice_difference'] = rank
    with path.open('w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def _write_summary_csv(path: Path, analysis: dict[str, Any]) -> None:
    fields = ['role', 'model', 'metric', 'image_macro_mean', 'mean_ci95_low', 'mean_ci95_high', 'baseline_model', 'reference_minus_baseline_mean_difference', 'difference_ci95_low', 'difference_ci95_high', 'reference_wins', 'ties', 'baseline_wins', 'wilcoxon_group_mean_statistic', 'wilcoxon_p_raw', 'wilcoxon_p_holm']
    model_lookup = {model['model']: model for model in analysis['models']}
    comparison_lookup = {item['baseline_model']: item for item in analysis['comparisons']}
    rows: list[dict[str, Any]] = []
    for model in analysis['models']:
        for metric in METRICS:
            ci_low, ci_high = model['group_bootstrap_ci95'][metric]
            rows.append({'role': model['role'], 'model': model['model'], 'metric': metric, 'image_macro_mean': model['image_macro_mean'][metric], 'mean_ci95_low': ci_low, 'mean_ci95_high': ci_high, 'baseline_model': '', 'reference_minus_baseline_mean_difference': '', 'difference_ci95_low': '', 'difference_ci95_high': '', 'reference_wins': '', 'ties': '', 'baseline_wins': '', 'wilcoxon_group_mean_statistic': '', 'wilcoxon_p_raw': '', 'wilcoxon_p_holm': ''})
    for baseline_model, comparison in comparison_lookup.items():
        for metric in METRICS:
            result = comparison['metrics'][metric]
            diff_low, diff_high = result['group_bootstrap_ci95']
            counts = result['case_wins_ties_losses']
            test = result['group_mean_signed_rank'] or {}
            rows.append({'role': 'comparison', 'model': analysis['reference_model'], 'metric': metric, 'image_macro_mean': model_lookup[analysis['reference_model']]['image_macro_mean'][metric], 'mean_ci95_low': model_lookup[analysis['reference_model']]['group_bootstrap_ci95'][metric][0], 'mean_ci95_high': model_lookup[analysis['reference_model']]['group_bootstrap_ci95'][metric][1], 'baseline_model': baseline_model, 'reference_minus_baseline_mean_difference': result['reference_minus_baseline_image_macro_mean'], 'difference_ci95_low': diff_low, 'difference_ci95_high': diff_high, 'reference_wins': counts['reference_wins'], 'ties': counts['ties'], 'baseline_wins': counts['baseline_wins'], 'wilcoxon_group_mean_statistic': test.get('statistic', ''), 'wilcoxon_p_raw': test.get('p_raw', ''), 'wilcoxon_p_holm': test.get('p_holm', '')})
    with path.open('w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def write_outputs(out_dir: Path, runs: list[Run], case_ids: list[str], analysis: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, run in enumerate(runs[1:]):
        comparison = analysis['comparisons'][index]
        pair_dir = out_dir / f'pair_{safe_slug(run.model)}'
        pair_dir.mkdir(parents=False, exist_ok=False)
        _write_ranked_cases(pair_dir / 'cases_ranked.csv', runs[0], run, case_ids)
        _plot_pair(out_dir, runs[0], run, case_ids, comparison)
    with (out_dir / 'summary.json').open('w', encoding='utf-8', newline='') as output:
        json.dump(_jsonable(analysis), output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')
    _write_summary_csv(out_dir / 'summary.csv', analysis)

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Compare paired segmentation evaluation directories using image-macro means and a group-cluster bootstrap.')
    parser.add_argument('--reference', required=True, type=Path, help='reference evaluation directory')
    parser.add_argument('--baselines', required=True, nargs='+', type=Path, help='one or more baseline evaluation directories')
    parser.add_argument('--out', required=True, type=Path, help='new or empty output directory')
    parser.add_argument('--resamples', type=int, default=10000, help='number of group bootstrap resamples (default: 10000)')
    parser.add_argument('--seed', type=int, default=20261105, help='bootstrap RNG seed (default: 20261105)')
    parser.add_argument('--wilcoxon', action='store_true', help='include descriptive group-mean signed-rank tests with Holm adjustment')
    return parser

def main(argv: list[str] | None=None) -> int:
    args = _parser().parse_args(argv)
    if args.resamples < 1:
        print('error: --resamples must be at least 1', file=sys.stderr)
        return 2
    try:
        runs = [load_run(args.reference), *(load_run(path) for path in args.baselines)]
        case_ids, group_labels = validate_runs(runs)
        out_dir = resolve_output_path(args.out, runs)
        import numpy
        import matplotlib
        if args.wilcoxon:
            import scipy.stats
        analysis = _make_analysis(runs, case_ids, group_labels, args)
        write_outputs(out_dir, runs, case_ids, analysis)
    except (OSError, ValueError, ImportError, ArithmeticError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
    print(f'Wrote benchmark comparison to {out_dir}')
    print(f"Models: {', '.join((run.model for run in runs))}; cases: {len(case_ids)}; groups: {len(group_labels)}")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
