import sys
from pathlib import Path
HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))
import csv, json, hashlib, shutil, time, os
import cv2
import numpy as np
from diagnostics import case_diagnostics
from evidence import write_inventory, validate_inventory
SOURCE = Path(os.environ['SCAR_CLASSICAL_SOURCE'])
OUT = Path(os.environ['SCAR_CLASSICAL_OUT'])

def read(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)

def predict(image, rule):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if rule['polarity'] == 'dark':
        mask = 255 - mask
    for op in rule['operations']:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN if op == 'open' else cv2.MORPH_CLOSE, np.ones((rule['kernel'], rule['kernel']), np.uint8))
    return mask > 0

def metrics(pred, gt):
    tp, fp = (int((pred & gt).sum()), int((pred & ~gt).sum()))
    fn, tn = (int((~pred & gt).sum()), int((~pred & ~gt).sum()))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, Dice=2 * tp / (2 * tp + fp + fn), IoU=tp / (tp + fp + fn), Precision=tp / (tp + fp) if tp + fp else float('nan'), Recall=tp / (tp + fn) if tp + fn else float('nan'))

def main():
    OUT.mkdir(exist_ok=True)
    for split in ('val', 'test'):
        validate_inventory(SOURCE / split)
    rules = [dict(polarity=p, operations=ops, kernel=k) for p in ('dark', 'bright') for ops, k in [((), 1), (('open',), 3), (('close',), 3), (('open', 'close'), 3), (('close', 'open'), 3), (('open',), 5), (('close',), 5), (('open', 'close'), 5)]]
    candidates = []
    for rule in rules:
        scores = []
        for image in sorted((SOURCE / 'val/case_images').glob('*.png')):
            gt = read(SOURCE / 'val/ground_truth' / image.name) > 0
            scores.append(metrics(predict(read(image), rule), gt)['Dice'])
        candidates.append(dict(rule=rule, validation_Dice=float(np.mean(scores))))
    best = max(candidates, key=lambda item: item['validation_Dice'])
    rule = best['rule']
    selection = dict(method='Otsu grayscale + validation-selected morphology', candidates=candidates, selected=best, selection_split='val', test_labels_used_for_selection=False, input='same unpadded 256 letterbox RGB evidence; grayscale conversion', code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    selection_bytes = json.dumps(selection, sort_keys=True, indent=2).encode()
    rule_path = OUT / 'selected_rule.json'
    if rule_path.exists() and rule_path.read_bytes() != selection_bytes:
        raise ValueError('Existing selected rule differs; preserve outputs and choose a new directory')
    rule_path.write_bytes(selection_bytes)
    for split in ('val', 'test'):
        src, out = (SOURCE / split, OUT / split)
        if (out / 'artifact_inventory.json').exists():
            validate_inventory(out)
            continue
        if out.exists():
            raise RuntimeError('Incomplete export preserved; archive it before retrying: ' + str(out))
        out.mkdir()
        for folder in ('predictions', 'scores_binary', 'ground_truth', 'case_images'):
            (out / folder).mkdir()
        original = list(csv.DictReader((src / 'per_case_metrics.csv').open(encoding='utf-8-sig')))
        rows, times = ([], [])
        for row in original:
            sid = row['sample_id']
            image = read(src / 'case_images' / (sid + '.png'))
            gt = read(src / 'ground_truth' / (sid + '.png')) > 0
            start = time.perf_counter()
            pred = predict(image, rule)
            times.append(time.perf_counter() - start)
            score = pred.astype(np.float32)
            rows.append(dict(sample_id=sid, group=row['group'], **metrics(pred, gt), **case_diagnostics(pred, gt, score)))
            cv2.imencode('.png', pred.astype(np.uint8) * 255)[1].tofile(str(out / 'predictions' / (sid + '.png')))
            np.save(out / 'scores_binary' / (sid + '.npy'), score)
            for folder in ('ground_truth', 'case_images'):
                shutil.copyfile(src / folder / (sid + '.png'), out / folder / (sid + '.png'))
        with (out / 'per_case_metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        meta = json.loads((src / 'evaluation.json').read_text(encoding='utf-8'))
        meta['protocol'] = dict(meta['protocol'], threshold='per-image grayscale Otsu; validation-locked polarity and morphology; binary decision', loss='not applicable; no neural training')
        meta.update(model='otsu_morphology', model_spec=selection, seed=None, checkpoint_sha256=hashlib.sha256(selection_bytes).hexdigest(), checkpoint_location=str(rule_path), runner_sha256=selection['code_sha256'], factory_sha256=None, retained_counts_matched=False, retained_checkpoint_provenance=None, probabilities_exported=False, score_note='Binary decisions only; scores are not calibrated probabilities; no continuous ROC claim', runtime_note='CPU OpenCV; neural CUDA timing not applicable', summary={k: dict(mean=float(np.nanmean([r[k] for r in rows])), sample_sd=float(np.nanstd([r[k] for r in rows], ddof=1)), defined_n=int(np.isfinite([r[k] for r in rows]).sum())) for k in ('Dice', 'IoU', 'HD95_network_px', 'ASSD_network_px', 'BoundaryF1_1px')})
        (out / 'evaluation.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        (out / 'cpu_timing.json').write_text(json.dumps(dict(seconds=times, mean_ms=1000 * float(np.mean(times)), protocol='CPU threshold and morphology only; excludes decoding; actual un-warmed sequence')), encoding='utf-8')
        for name in ('case_geometry.json', 'dataset_lock.json'):
            shutil.copyfile(src / name, out / name)
        shutil.copyfile(__file__, out / 'run_classical.py')
        shutil.copyfile(rule_path, out / 'selected_rule.json')
        write_inventory(out)
        print(split, meta['summary'], flush=True)
if __name__ == '__main__':
    main()
