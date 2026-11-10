import argparse
import os
from pathlib import Path
import runpy

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Parent of exported val/ and test/ evaluation folders')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Choose a new output directory')
    os.environ['SCAR_CLASSICAL_SOURCE'] = str(a.source.resolve())
    os.environ['SCAR_CLASSICAL_OUT'] = str(a.out.resolve())
    script = Path(__file__).resolve().parents[1] / 'benchmark/extended_baselines/run_classical.py'
    runpy.run_path(str(script), run_name='__main__')
if __name__ == '__main__':
    main()
