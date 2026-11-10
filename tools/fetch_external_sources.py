import argparse
import json
from pathlib import Path
import subprocess

def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs='+', choices=['TransUNet', 'PANet', 'SAMed'], required=True)
    args = p.parse_args()
    specs = json.loads((root / 'external_sources.json').read_text(encoding='utf-8'))
    for spec in specs:
        if spec['name'] not in args.models:
            continue
        destination = root / spec['destination']
        if destination.exists():
            current = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
            if current != spec['commit']:
                raise ValueError('Existing checkout has a different commit: ' + str(destination))
            print(spec['name'], 'already at', current)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', spec['repository'], str(destination)], check=True)
        subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', spec['commit']], check=True)
        print(spec['name'], spec['commit'])
if __name__ == '__main__':
    main()
