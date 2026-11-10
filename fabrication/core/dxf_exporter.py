import traceback
from typing import List
from core.contour import Contour

class DXFExporter:

    @staticmethod
    def export_to_dxf(contours: List[Contour], pixels_per_cm: float, filename: str, label_font_size_mm: float=5.0, label_min_size_mm: float=5.0):
        try:
            import ezdxf
            from ezdxf.enums import TextEntityAlignment
        except ImportError:
            raise ImportError('DXF export requires ezdxf. Install with: pip install ezdxf')
        SCALE_FACTOR = 1.0 / 2.3
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        doc.layers.add('CONTOURS', color=1)
        doc.layers.add('DIMENSIONS', color=5)
        doc.layers.add('LABELS', color=7)
        doc.header['$INSUNITS'] = 4
        print(f'\n=== DXF export ===')
        print(f'Pixel scale: {pixels_per_cm} pixels/cm')
        print(f'Number of contours: {len(contours)}')
        print(f'Label font size: {label_font_size_mm}mm, minimum size threshold: {label_min_size_mm}mm')
        print(f'Global scale factor: {SCALE_FACTOR:.3f} (1/2.3)')
        for i, contour in enumerate(contours):
            if not contour.nurbs_points:
                print(f'Contour {i} has no NURBS points; skipping')
                continue
            display_rect = contour.get_display_rect()
            print(f'Contour {i} (label {contour.label}):')
            print(f'  Canvas position: ({display_rect.x():.1f}, {display_rect.y():.1f}) pixels')
            print(f'  Canvas dimensions: {display_rect.width():.1f}x{display_rect.height():.1f} pixels')
            print(f'  Contour scale: {contour.scale:.3f}')
            print(f'  NURBS points: {len(contour.nurbs_points)}')
            points = []
            scale = contour.scale
            bbox = contour.bounding_box
            for j, nurbs_point in enumerate(contour.nurbs_points):
                x_px = display_rect.left() + (nurbs_point.x() - bbox.left()) * scale
                y_px = display_rect.top() + (nurbs_point.y() - bbox.top()) * scale
                x_mm = x_px * 10 / pixels_per_cm * SCALE_FACTOR
                y_mm = y_px * 10 / pixels_per_cm * SCALE_FACTOR
                y_mm = 150 * SCALE_FACTOR - y_mm
                points.append((x_mm, y_mm))
                if j < 3:
                    print(f'    Point {j}: local({nurbs_point.x():.1f}, {nurbs_point.y():.1f}) -> canvas({x_px:.1f}, {y_px:.1f}) -> scaled DXF({x_mm:.1f}, {y_mm:.1f})')
            if len(points) >= 2:
                if points[0] != points[-1]:
                    points.append(points[0])
                msp.add_lwpolyline(points, dxfattribs={'layer': 'CONTOURS', 'closed': True, 'lineweight': 5})
                print(f'  Exported: {len(points)} points')
            if contour.label > 0:
                label_pos, dist_px = contour.get_label_position(pixels_per_cm=pixels_per_cm, font_size_mm=label_font_size_mm, min_size_mm=label_min_size_mm)
                if label_pos is not None and dist_px > 0:
                    label_x_mm = label_pos.x() * 10 / pixels_per_cm * SCALE_FACTOR
                    label_y_mm = (150 - label_pos.y() * 10 / pixels_per_cm) * SCALE_FACTOR
                    text = msp.add_text(str(contour.label), dxfattribs={'layer': 'LABELS', 'height': label_font_size_mm * SCALE_FACTOR, 'color': 7})
                    text.set_placement((label_x_mm, label_y_mm), align=TextEntityAlignment.MIDDLE_CENTER)
                    print(f"  Label '{contour.label}' added at scaled position ({label_x_mm:.1f}, {label_y_mm:.1f})mm")
                else:
                    print(f'  Contour {contour.label} has no suitable label position; skipping')
            center_x = (display_rect.left() + display_rect.width() / 2) * 10 / pixels_per_cm * SCALE_FACTOR
            center_y = (150 - (display_rect.top() + display_rect.height() / 2) * 10 / pixels_per_cm) * SCALE_FACTOR
            msp.add_point((center_x, center_y), dxfattribs={'layer': 'DIMENSIONS'})
        doc.saveas(filename)
        print(f'\nExported {len([c for c in contours if c.nurbs_points])} contours to {filename}')
        print(f'Scaled DXF dimensions: approximately {150 * SCALE_FACTOR:.1f}mm x {150 * SCALE_FACTOR:.1f}mm (original 150 mm)')
        return True
