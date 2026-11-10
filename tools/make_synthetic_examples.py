import argparse
import json
import math
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Choose a new output directory')
    from PIL import Image, ImageDraw
    a.out.mkdir(parents=True)
    (a.out / 'contours').mkdir()
    image = Image.new('L', (256, 256), 0)
    draw = ImageDraw.Draw(image)
    draw.ellipse((25, 30, 100, 105), fill=255)
    draw.polygon([(135, 35), (205, 45), (190, 70), (215, 105), (160, 125), (125, 85)], fill=255)
    draw.ellipse((45, 170, 210, 205), fill=255)
    image.save(a.out / 'mask.png')
    for i, (rx, ry) in enumerate([(4, 4), (7, 3), (10, 6)], 1):
        points = [[rx * math.cos(t * 2 * math.pi / 160), ry * math.sin(t * 2 * math.pi / 160)] for t in range(160)]
        (a.out / 'contours' / f'example_{i:02d}.json').write_text(json.dumps({'points_mm': points}, indent=2), encoding='utf-8')
    print('Created synthetic examples in', a.out)
if __name__ == '__main__':
    main()
