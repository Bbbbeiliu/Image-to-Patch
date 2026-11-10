import argparse
import json
from pathlib import Path
import sys

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--diameter-mm', type=float, default=10.0)
    p.add_argument('--yolo-model', type=Path, default=None)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Choose a new output directory')
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from core.marker_detector import WhiteBallMarkerDetector
    a.out.mkdir(parents=True)
    detector = WhiteBallMarkerDetector(a.diameter_mm, str(a.yolo_model) if a.yolo_model else None)
    result = detector.process_single_image(str(a.image), str(a.out))
    summary = {key: value for key, value in result.items() if key != 'comparison_image'}
    (a.out / 'calibration.json').write_text(json.dumps(summary, indent=2, default=float), encoding='utf-8')
    print(json.dumps(summary, indent=2, default=float))
if __name__ == '__main__':
    main()
