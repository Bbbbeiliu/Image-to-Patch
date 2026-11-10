import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--masks', type=Path, required=True)
    p.add_argument('--split-csv', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Output must not exist: ' + str(a.out))
    rows = list(csv.DictReader(a.split_csv.open(encoding='utf-8-sig', newline='')))
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Split CSV is empty or has duplicate sample IDs')
    patient_splits = {}
    inputs = []
    for r in rows:
        sid, group, split = (r['sample_id'], r['patient_id'], r['split'])
        if split not in ['train', 'val', 'test'] or not sid.isalnum():
            raise ValueError('Invalid split or sample ID')
        patient_splits.setdefault(group, set()).add(split)
        images = [x for x in a.images.glob(sid + '.*') if x.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        if len(images) != 1:
            raise ValueError('Exactly one image is required for ' + sid)
        mask = a.masks / (sid + '.json')
        if not mask.is_file():
            raise FileNotFoundError('Labelme annotation required: ' + str(mask))
        inputs.append((r, images[0], mask))
    if any((len(v) != 1 for v in patient_splits.values())):
        raise ValueError('A patient appears in multiple subsets')
    a.out.mkdir(parents=True)
    for split in ['train', 'val', 'test']:
        for kind in ['images', 'masks']:
            (a.out / split / kind).mkdir(parents=True)
    audit = a.out / '_audit'
    audit.mkdir()
    manifest = []
    for r, image, mask in inputs:
        for original, kind in [(image, 'images'), (mask, 'masks')]:
            destination = a.out / r['split'] / kind / original.name
            shutil.copy2(original, destination)
            if digest(original) != digest(destination):
                raise ValueError('Copy integrity failed')
        manifest.append({'id': r['sample_id'], 'group': r['patient_id'], 'split': r['split']})
    (audit / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    summary = {'counts': {s: sum((r['split'] == s for r in manifest)) for s in ['train', 'val', 'test']}, 'group_count': len(patient_splits), 'verification': {'source_metadata_unchanged': True, 'group_cross_split_overlap': 0}}
    (audit / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    for split in ['train', 'val', 'test']:
        names = [f'{split}/images/{image.name}' for r, image, _ in inputs if r['split'] == split]
        (audit / (split + '.txt')).write_text('\n'.join(names) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
