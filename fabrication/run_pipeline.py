import argparse
import json
import os
from pathlib import Path
import sys

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mask', type=Path, required=True)
    p.add_argument('--mm-per-pixel', type=float, required=True)
    p.add_argument('--control-points', type=int, default=120)
    p.add_argument('--font-mm', type=float, default=3.0)
    p.add_argument('--min-label-mm', type=float, default=3.0)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.mm_per_pixel <= 0:
        p.error('--mm-per-pixel must be positive')
    if a.out.exists():
        raise FileExistsError('Choose a new output directory')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    home = Path(__file__).resolve().parent
    sys.path.insert(0, str(home))
    sys.path.insert(0, str(home / 'nesting'))
    import numpy as np
    from core.image_processor import AdvancedImageProcessor
    from core.contour import Contour
    from core.dxf_exporter import DXFExporter
    import comparison_methods as nesting
    a.out.mkdir(parents=True)
    fitted, nesting_inputs = ({}, [])
    for i, (points, _) in enumerate(AdvancedImageProcessor.load_and_process_image(str(a.mask)), 1):
        contour = Contour(points, i, source_image=a.mask.name, label=i)
        contour.nurbs_points, contour.nurbs_curve = AdvancedImageProcessor.smooth_contour_with_nurbs(points, num_control_points=a.control_points)
        if len(contour.nurbs_points) < 3:
            continue
        xy = [[point.x(), point.y()] for point in contour.nurbs_points]
        contour.actual_width_cm = contour.bounding_box.width() * a.mm_per_pixel / 10
        contour.actual_height_cm = contour.bounding_box.height() * a.mm_per_pixel / 10
        contour.pixel_scale_mm_per_px = a.mm_per_pixel
        fitted[i] = contour
        nesting_inputs.append(nesting.SimpleContour(i, xy, contour.actual_width_cm, contour.actual_height_cm, a.mask.name))
    if not nesting_inputs:
        raise ValueError('No valid contours were extracted')
    pages = nesting.polar_greedy_nesting(nesting_inputs, nesting.MARGIN_PX)
    rows = []
    for page_index, page in enumerate(pages, 1):
        export = []
        for placed in page:
            contour = fitted[placed.id]
            contour.position = placed.position
            contour.scale = placed.scale
            contour.scale_x, contour.scale_y = (placed.scale_x, placed.scale_y)
            position, span = contour.get_label_position(nesting.PIXELS_PER_CM, a.font_mm, a.min_label_mm)
            rows.append({'contour_id': contour.id, 'page': page_index, 'label_placed': position is not None, 'label_position_px': [position.x(), position.y()] if position else None, 'label_span_px': span})
            export.append(contour)
        target = a.out / f'page_{page_index:02d}.dxf'
        DXFExporter.export_to_dxf(export, nesting.PIXELS_PER_CM, str(target), a.font_mm, a.min_label_mm)
        if not target.is_file():
            raise RuntimeError('DXF export failed')
    (a.out / 'summary.json').write_text(json.dumps({'input_mask': a.mask.name, 'mm_per_pixel': a.mm_per_pixel, 'input_contours': len(fitted), 'pages': len(pages), 'working_radius_cm': nesting.CONTAINER_RADIUS_CM, 'spacing_mm': 1.0, 'dxf_display_scale_factor': 1.0 / 2.3, 'contours': rows}, indent=2), encoding='utf-8')
    print('Exported', len(pages), 'DXF page(s) to', a.out)
if __name__ == '__main__':
    main()
