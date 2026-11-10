import ast
import csv
import hashlib
import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    python_files = [p for p in root.rglob('*.py') if not any((x in p.parts for x in ['vendor', 'torch_cache']))]
    for p in python_files:
        compile(p.read_text(encoding='utf-8-sig'), str(p), 'exec')
    for p in (root / 'notebooks').glob('*.ipynb'):
        notebook = json.loads(p.read_text(encoding='utf-8'))
        assert notebook['nbformat'] == 4
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), str(p), 'exec')
    for p in (root / 'experiments/splits').glob('*.csv'):
        rows = list(csv.DictReader(p.open(encoding='utf-8')))
        assert len({r['sample_id'] for r in rows}) == len(rows)
        groups = {}
        for r in rows:
            groups.setdefault(r['patient_id'], set()).add(r['split'])
        assert all((len(v) == 1 for v in groups.values())), p
    checksum_file = root / 'SHA256SUMS.txt'
    if checksum_file.exists():
        for line in checksum_file.read_text(encoding='utf-8').splitlines():
            expected, relative = line.split('  ', 1)
            assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative
    print(json.dumps({'status': 'passed', 'python_modules': len(python_files), 'notebooks': len(list((root / 'notebooks').glob('*.ipynb')))}, indent=2))
if __name__ == '__main__':
    main()
