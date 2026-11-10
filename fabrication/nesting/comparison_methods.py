import json, csv, math, time, os, copy
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtCore import QPointF, QRectF
from shapely.geometry import Polygon, Point, box
from shapely import affinity
from shapely.ops import unary_union
from shapely.prepared import prep
from rectpack import newPacker, PackingMode
from tqdm import tqdm
import nfp_utls
PIXELS_PER_CM = 75.59
CONTAINER_RADIUS_CM = 5.0
CONTAINER_RADIUS_PX = CONTAINER_RADIUS_CM * PIXELS_PER_CM
CENTER = (CONTAINER_RADIUS_PX, CONTAINER_RADIUS_PX)
MARGIN_PX = 1.0 * (PIXELS_PER_CM / 10)
OUTPUT_DIR = Path('comparison_results')
DATASET_SIZES = [40, 80, 100, 130]
DATASETS = {f'Dataset_{s}': f'datasets/Dataset_{s}' for s in DATASET_SIZES}

class SimpleContour:

    def __init__(self, cid: int, nurbs_points: List[List[float]], width_cm: float, height_cm: float, mask: str):
        self.id = cid
        self.label = cid
        self.nurbs_points_raw = nurbs_points
        self.position = QPointF(0, 0)
        self.scale_x = self.scale_y = 1.0
        self.final_polygon = None
        xs = [p[0] for p in nurbs_points]
        ys = [p[1] for p in nurbs_points]
        self.bounding_box = QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        if self.bounding_box.width() > 0 and self.bounding_box.height() > 0:
            self.scale_x = width_cm * PIXELS_PER_CM / self.bounding_box.width()
            self.scale_y = height_cm * PIXELS_PER_CM / self.bounding_box.height()
        self.scale = (self.scale_x + self.scale_y) / 2.0

    def get_display_rect(self) -> QRectF:
        w = self.bounding_box.width() * self.scale_x
        h = self.bounding_box.height() * self.scale_y
        return QRectF(self.position.x(), self.position.y(), w, h)

    def get_display_polygon(self) -> Polygon:
        rect = self.get_display_rect()
        bbox = self.bounding_box
        pts = [(rect.x() + (p[0] - bbox.x()) * self.scale_x, rect.y() + (p[1] - bbox.y()) * self.scale_y) for p in self.nurbs_points_raw]
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly

def load_dataset(dataset_dir: str) -> List[SimpleContour]:
    json_dir = Path(dataset_dir)
    csv_path = json_dir / 'contour_info.csv'
    if not csv_path.exists():
        return []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    row_map = {int(r['contour_id']): r for r in rows}
    contours = []
    for cid in row_map:
        jf = json_dir / f'contour_{cid:03d}.json'
        if not jf.exists():
            continue
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cont = SimpleContour(cid, data['nurbs_points'], data['actual_width_cm'], data['actual_height_cm'], data.get('source_mask', ''))
        contours.append(cont)
    contours.sort(key=lambda c: c.bounding_box.width() * c.bounding_box.height(), reverse=True)
    return contours

def poly_in_circle(poly, cx, cy, radius):
    if poly.is_empty:
        return False
    minx, miny, maxx, maxy = poly.bounds
    if (maxx - cx) ** 2 + (maxy - cy) ** 2 > radius ** 2:
        return False
    for x, y in poly.exterior.coords:
        if (x - cx) ** 2 + (y - cy) ** 2 > radius ** 2:
            return False
    return True

def try_place_optimized(poly, page_data, container_radius, center):
    poly_bounds = poly.bounds
    poly_width = poly_bounds[2] - poly_bounds[0]
    poly_height = poly_bounds[3] - poly_bounds[1]
    max_dim = max(poly_width, poly_height)
    step_r = max_dim / 2
    radii = [i * step_r for i in range(int(container_radius / step_r) + 1)]
    placed_prep = page_data['prep']
    for r in radii:
        if r == 0:
            angles = [0.0]
        else:
            circumference = 2 * math.pi * r
            n_angles = max(1, int(circumference / (max_dim / 2)))
            n_angles = min(n_angles, 36)
            angles = [2 * math.pi * i / n_angles for i in range(n_angles)]
        for angle in angles:
            x = center[0] + r * math.cos(angle)
            y = center[1] + r * math.sin(angle)
            dx = x - (poly_bounds[0] + poly_bounds[2]) / 2
            dy = y - (poly_bounds[1] + poly_bounds[3]) / 2
            candidate = affinity.translate(poly, dx, dy)
            if not poly_in_circle(candidate, center[0], center[1], container_radius):
                continue
            if placed_prep is not None and placed_prep.intersects(candidate):
                continue
            return (dx, dy, candidate)
    return None

def polar_greedy_nesting(contours: List[SimpleContour], margin_px: float) -> List[List[SimpleContour]]:
    contours_data = []
    for c in contours:
        rect = c.get_display_rect()
        if rect.isNull():
            continue
        pts = []
        scale = c.scale
        bbox = c.bounding_box
        for p in c.nurbs_points_raw:
            x_px = rect.x() + (p[0] - bbox.x()) * scale
            y_px = rect.y() + (p[1] - bbox.y()) * scale
            pts.append((x_px, y_px))
        orig_poly = Polygon(pts)
        if not orig_poly.is_valid:
            orig_poly = orig_poly.buffer(0)
        poly_margin = orig_poly.buffer(margin_px, join_style=2) if margin_px > 0 else orig_poly
        if not poly_margin.is_valid:
            poly_margin = poly_margin.buffer(0)
        contours_data.append({'contour': c, 'original_poly': orig_poly, 'poly_with_margin': poly_margin})
    contours_data.sort(key=lambda x: x['poly_with_margin'].area, reverse=True)
    pages = []
    for data in tqdm(contours_data, desc='PolarGreedy', leave=False):
        placed = False
        for page in pages:
            res = try_place_optimized(data['poly_with_margin'], page, CONTAINER_RADIUS_PX, CENTER)
            if res:
                dx, dy, new_margin_poly = res
                data['original_poly'] = affinity.translate(data['original_poly'], dx, dy)
                data['poly_with_margin'] = new_margin_poly
                data['contour'].position += QPointF(dx, dy)
                data['contour'].final_polygon = data['original_poly']
                page['items'].append(data)
                if page['union'] is None:
                    page['union'] = data['poly_with_margin']
                else:
                    page['union'] = unary_union([page['union'], data['poly_with_margin']])
                page['prep'] = prep(page['union'])
                placed = True
                break
        if not placed:
            res = try_place_optimized(data['poly_with_margin'], {'items': [], 'union': None, 'prep': None}, CONTAINER_RADIUS_PX, CENTER)
            if res:
                dx, dy, new_margin_poly = res
                data['original_poly'] = affinity.translate(data['original_poly'], dx, dy)
                data['poly_with_margin'] = new_margin_poly
                data['contour'].position += QPointF(dx, dy)
                data['contour'].final_polygon = data['original_poly']
                new_page = {'items': [data], 'union': data['poly_with_margin'], 'prep': prep(data['poly_with_margin'])}
                pages.append(new_page)
            else:
                print(f"Warning: contour {data['contour'].id} cannot be placed.")
    return [[item['contour'] for item in page['items']] for page in pages]

def rect_pack_nesting(contours: List[SimpleContour], margin_px: float) -> List[List[SimpleContour]]:
    _MARGIN = 0.0
    items_data = []
    for c in contours:
        rect = c.get_display_rect()
        if rect.isNull():
            continue
        w = rect.width()
        h = rect.height()
        orig_poly = c.get_display_polygon()
        if orig_poly.is_empty:
            continue

        def make_rect_poly(ww, hh):
            return box(0, 0, ww, hh)
        candidates = []
        rect0 = make_rect_poly(w, h)
        candidates.append((rect0, w, h, orig_poly))
        if abs(w - h) > 1e-06:
            rotated_poly = affinity.rotate(orig_poly, 90, origin=(0, 0))
            if not rotated_poly.is_valid:
                rotated_poly = rotated_poly.buffer(0)
            minx, miny, _, _ = rotated_poly.bounds
            if minx < 0 or miny < 0:
                rotated_poly = affinity.translate(rotated_poly, -minx, -miny)
            rect1 = make_rect_poly(h, w)
            candidates.append((rect1, h, w, rotated_poly))
        items_data.append({'contour': c, 'area': orig_poly.area, 'candidates': candidates})
    items_data.sort(key=lambda x: x['area'], reverse=True)
    pages = []
    for data in tqdm(items_data, desc='RectPack', leave=False):
        c = data['contour']
        placed = False
        for page_data in pages:
            for margin_poly, orig_w, orig_h, transformed_poly in data['candidates']:
                res = try_place_optimized(margin_poly, page_data, CONTAINER_RADIUS_PX, CENTER)
                if res:
                    dx, dy, new_margin_poly = res
                    new_orig_poly = affinity.translate(transformed_poly, dx, dy)
                    c.position += QPointF(dx, dy)
                    c.final_polygon = new_orig_poly
                    page_data['items'].append({'contour': c, 'original_poly': new_orig_poly, 'poly_without_margin': new_margin_poly})
                    if page_data['union'] is None:
                        page_data['union'] = new_margin_poly
                    else:
                        page_data['union'] = unary_union([page_data['union'], new_margin_poly])
                    page_data['prep'] = prep(page_data['union'])
                    placed = True
                    break
            if placed:
                break
        if not placed:
            for margin_poly, orig_w, orig_h, transformed_poly in data['candidates']:
                new_page_data = {'items': [], 'union': None, 'prep': None}
                res = try_place_optimized(margin_poly, new_page_data, CONTAINER_RADIUS_PX, CENTER)
                if res:
                    dx, dy, new_margin_poly = res
                    new_orig_poly = affinity.translate(transformed_poly, dx, dy)
                    c.position += QPointF(dx, dy)
                    c.final_polygon = new_orig_poly
                    new_page_data['items'].append({'contour': c, 'original_poly': new_orig_poly, 'poly_without_margin': new_margin_poly})
                    new_page_data['union'] = new_margin_poly
                    new_page_data['prep'] = prep(new_margin_poly)
                    pages.append(new_page_data)
                    placed = True
                    break
            if not placed:
                print(f'Warning: rectangular contour {c.id} cannot be placed in circle.')
    result_pages = []
    for page_data in pages:
        result_pages.append([item['contour'] for item in page_data['items']])
    return result_pages
NFP_CACHE = {}

def polygon_to_nfp_format(poly: Polygon) -> Dict:
    coords = list(poly.exterior.coords)[:-1]
    points = [{'x': x, 'y': y} for x, y in coords]
    return {'points': points}

def get_nfp_shape(fixed: Polygon, moving: Polygon, id_a: int, id_b: int) -> Optional[Polygon]:
    key = (id_a, id_b) if id_a < id_b else (id_b, id_a)
    if key not in NFP_CACHE:
        A = polygon_to_nfp_format(fixed)
        B = polygon_to_nfp_format(moving)
        try:
            nfp_list = nfp_utls.nfp_polygon(copy.deepcopy(A), copy.deepcopy(B), inside=False)
            if nfp_list and nfp_list[0]:
                pts = [(p['x'], p['y']) for p in nfp_list[0]]
                NFP_CACHE[key] = Polygon(pts)
            else:
                NFP_CACHE[key] = None
        except:
            NFP_CACHE[key] = None
    return NFP_CACHE[key]

def nfp_overlap(fixed: Polygon, moving: Polygon, id_a: int=None, id_b: int=None) -> bool:
    if id_a is not None and id_b is not None:
        nfp_shape = get_nfp_shape(fixed, moving, id_a, id_b)
        if nfp_shape is None:
            return fixed.intersects(moving)
        ref = moving.centroid
        return nfp_shape.intersects(ref) or nfp_shape.contains(ref)
    else:
        return fixed.intersects(moving)

def try_place_nfp(poly_margin, placed_polys, margin, cont_id=None):
    if placed_polys:
        placed_union = unary_union(placed_polys)
        placed_prep = prep(placed_union)
    else:
        placed_prep = None
    max_dim = max(poly_margin.bounds[2] - poly_margin.bounds[0], poly_margin.bounds[3] - poly_margin.bounds[1])
    step_r = max_dim / 2
    radii = [i * step_r for i in range(int(CONTAINER_RADIUS_PX / step_r) + 1)]
    for r in radii:
        if r == 0:
            angles = [0.0]
        else:
            n_angles = max(1, int(2 * math.pi * r / (max_dim / 2)))
            angles = [2 * math.pi * i / n_angles for i in range(n_angles)]
        for ang in angles:
            cx = CENTER[0] + r * math.cos(ang)
            cy = CENTER[1] + r * math.sin(ang)
            dx = cx - poly_margin.centroid.x
            dy = cy - poly_margin.centroid.y
            cand = affinity.translate(poly_margin, dx, dy)
            if not poly_in_circle(cand, CENTER[0], CENTER[1], CONTAINER_RADIUS_PX):
                continue
            if placed_prep is not None and placed_prep.intersects(cand):
                overlap = False
                for fp in placed_polys:
                    if cand.intersects(fp):
                        if nfp_overlap(fp, cand, id_a=None, id_b=cont_id):
                            overlap = True
                            break
                if overlap:
                    continue
            return (dx, dy, cand)
    return None

def nfp_nesting(contours, margin_px):
    NFP_CACHE.clear()
    pages = []
    sorted_contours = sorted(contours, key=lambda c: c.bounding_box.width() * c.bounding_box.height(), reverse=True)
    for c in tqdm(sorted_contours, desc='NFPPlacer', leave=False):
        c.position = QPointF(0, 0)
        poly = c.get_display_polygon()
        if poly.is_empty:
            continue
        pm = poly.buffer(margin_px, join_style=2)
        if not pm.is_valid:
            pm = pm.buffer(0)
        placed = False
        for page in pages:
            fixed_polys = [pc.final_polygon for pc in page if pc.final_polygon]
            res = try_place_nfp(pm, fixed_polys, margin_px, cont_id=c.id)
            if res:
                dx, dy, _ = res
                c.position += QPointF(dx, dy)
                c.final_polygon = affinity.translate(poly, dx, dy)
                page.append(c)
                placed = True
                break
        if not placed:
            res = try_place_nfp(pm, [], margin_px, cont_id=c.id)
            if res:
                dx, dy, _ = res
                c.position += QPointF(dx, dy)
                c.final_polygon = affinity.translate(poly, dx, dy)
                pages.append([c])
    return pages

def compute_avg_util(pages):
    total_area = 0.0
    for page in pages:
        for c in page:
            if c.final_polygon:
                total_area += c.final_polygon.area
            else:
                rect = c.get_display_rect()
                total_area += rect.width() * rect.height()
    total_page = len(pages) * math.pi * CONTAINER_RADIUS_PX ** 2
    return total_area / total_page if total_page > 0 else 0

def save_first_page(pages, method_name, dataset_name):
    if not pages or not pages[0]:
        return
    page = pages[0]
    fig, ax = plt.subplots(figsize=(8, 8))
    circle = plt.Circle(CENTER, CONTAINER_RADIUS_PX, fill=False, ec='black', lw=2)
    ax.add_artist(circle)
    for c in page:
        poly = c.final_polygon
        if poly:
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.4, ec='blue')
        else:
            rect = c.get_display_rect()
            ax.add_patch(plt.Rectangle((rect.x(), rect.y()), rect.width(), rect.height(), fill=None, ec='red', ls='--'))
    ax.set_xlim(CENTER[0] - CONTAINER_RADIUS_PX - 10, CENTER[0] + CONTAINER_RADIUS_PX + 10)
    ax.set_ylim(CENTER[1] - CONTAINER_RADIUS_PX - 10, CENTER[1] + CONTAINER_RADIUS_PX + 10)
    ax.set_aspect('equal')
    ax.set_title(f'{method_name} - {dataset_name}')
    plt.savefig(OUTPUT_DIR / f'{dataset_name}_{method_name}_firstpage.png', dpi=150)
    plt.close()

def main():
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    datasets = DATASETS
    methods = {'PolarGreedy': polar_greedy_nesting, 'RectPack': rect_pack_nesting, 'NFPPlacer': nfp_nesting}
    results = {ds: {} for ds in datasets}
    for ds_name, ds_path in datasets.items():
        print(f"\n{'=' * 40}\n  {ds_name}\n{'=' * 40}")
        contours = load_dataset(ds_path)
        if not contours:
            continue
        for m_name, m_func in methods.items():
            for c in contours:
                c.position = QPointF(0, 0)
                c.final_polygon = None
            start = time.perf_counter()
            pages = m_func(contours, MARGIN_PX)
            elapsed = time.perf_counter() - start
            util = compute_avg_util(pages) if pages else 0
            results[ds_name][m_name] = {'time': elapsed, 'pages': len(pages), 'utilization': util}
            print(f'  {m_name:15s}: time={elapsed:6.2f}s, pages={len(pages)}, util={util:.4f}')
            if pages:
                save_first_page(pages, m_name, ds_name)
    csv_file = OUTPUT_DIR / 'comparison_results.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Dataset', 'Method', 'Time(s)', 'Pages', 'Utilization'])
        for ds, mres in results.items():
            for m, s in mres.items():
                w.writerow([ds, m, s['time'], s['pages'], s['utilization']])
    md_file = OUTPUT_DIR / 'comparison_results.md'
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write('| Dataset | Method | Time(s) | Pages | Utilization |\n')
        f.write('|---------|--------|---------|-------|-------------|\n')
        for ds, mres in results.items():
            for m, s in mres.items():
                f.write(f"| {ds} | {m} | {s['time']:.2f} | {s['pages']} | {s['utilization']:.4f} |\n")
    datasets_names = [f'Dataset_{s}' for s in DATASET_SIZES]
    method_names = ['RectPack', 'PolarGreedy', 'NFPPlacer']
    colors = ['#D4A373', '#7A9E9F', '#B5C99A']
    x = np.arange(len(datasets_names))
    width = 0.25
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for i, method in enumerate(method_names):
        times = [results[ds][method]['time'] for ds in datasets_names]
        bars = ax1.bar(x + i * width, times, width, color=colors[i], edgecolor='white', linewidth=0.6, label=method)
        for bar, t in zip(bars, times):
            ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.1, f'{t:.1f}s', ha='center', va='bottom', fontsize=9)
    ax1.set_ylabel('Time (s)', fontsize=12)
    ax1.set_title('Computation Time', fontsize=14, weight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(datasets_names, fontsize=11)
    ax1.legend(frameon=False)
    ax1.grid(axis='y', linestyle=':', alpha=0.4)
    for i, method in enumerate(method_names):
        utils = [results[ds][method]['utilization'] for ds in datasets_names]
        bars = ax2.bar(x + i * width, utils, width, color=colors[i], edgecolor='white', linewidth=0.6, label=method)
        for bar, u in zip(bars, utils):
            ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.005, f'{u:.3f}', ha='center', va='bottom', fontsize=9)
    ax2.set_ylabel('Utilization', fontsize=12)
    ax2.set_title('Material Utilization', fontsize=14, weight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(datasets_names, fontsize=11)
    ax2.legend(frameon=False)
    ax2.grid(axis='y', linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comparison_bar_chart.pdf', dpi=300)
    plt.savefig(OUTPUT_DIR / 'comparison_bar_chart.png', dpi=300)
    plt.savefig(OUTPUT_DIR / 'comparison_bar_chart.svg', dpi=300, bbox_inches='tight')
    plt.show()
    print(f'\nDone. Results saved in {OUTPUT_DIR}')
    print(f' - CSV summary: {csv_file}')
    print(f' - Markdown table: {md_file}')
    print(f' - Figures: comparison_bar_chart.{{pdf,png}}, and first-page images per method.')
