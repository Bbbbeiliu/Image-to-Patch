import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import re

class FileUtils:

    @staticmethod
    def find_matching_mask(overlay_path: str, mask_dir: str) -> Optional[str]:
        try:
            overlay_name = Path(overlay_path).name
            if '_overlay' in overlay_name:
                prefix = overlay_name.split('_overlay')[0]
                mask_name = f'{prefix}_mask.png'
                mask_path = Path(mask_dir) / mask_name
                if mask_path.exists():
                    return str(mask_path)
                else:
                    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                        mask_path = Path(mask_dir) / f'{prefix}_mask{ext}'
                        if mask_path.exists():
                            return str(mask_path)
            return None
        except Exception as e:
            print(f'Error locating segmentation file: {e}')
            return None

    @staticmethod
    def find_matching_overlay(mask_path: str, overlay_dir: str) -> Optional[str]:
        try:
            mask_name = Path(mask_path).name
            if '_mask' in mask_name:
                prefix = mask_name.split('_mask')[0]
                overlay_name = f'{prefix}_overlay.png'
                overlay_path = Path(overlay_dir) / overlay_name
                if overlay_path.exists():
                    return str(overlay_path)
                else:
                    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                        overlay_path = Path(overlay_dir) / f'{prefix}_overlay{ext}'
                        if overlay_path.exists():
                            return str(overlay_path)
            return None
        except Exception as e:
            print(f'Error locating source image: {e}')
            return None

    @staticmethod
    def validate_image_pair(overlay_path: str, mask_path: str) -> bool:
        try:
            overlay_name = Path(overlay_path).stem
            mask_name = Path(mask_path).stem
            overlay_prefix = overlay_name.replace('_overlay', '')
            mask_prefix = mask_name.replace('_mask', '')
            return overlay_prefix == mask_prefix
        except Exception:
            return False

    @staticmethod
    def scan_image_pairs(overlay_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
        image_pairs = []
        try:
            overlay_dir_path = Path(overlay_dir)
            mask_dir_path = Path(mask_dir)
            if not overlay_dir_path.exists():
                print(f'Source image directory does not exist: {overlay_dir}')
                return image_pairs
            if not mask_dir_path.exists():
                print(f'Segmentation directory does not exist: {mask_dir}')
                return image_pairs
            overlay_patterns = ['*_overlay.png', '*_overlay.jpg', '*_overlay.jpeg', '*_overlay.bmp', '*_overlay.tif', '*_overlay.tiff']
            overlay_files = []
            for pattern in overlay_patterns:
                overlay_files.extend(list(overlay_dir_path.glob(pattern)))
            for overlay_file in overlay_files:
                mask_path = FileUtils.find_matching_mask(str(overlay_file), mask_dir)
                if mask_path and Path(mask_path).exists():
                    image_pairs.append((str(overlay_file), mask_path))
                else:
                    print(f'No matching segmentation found for {overlay_file.name} matching segmentation file')
            print(f'Found {len(image_pairs)} matched image pairs')
            return image_pairs
        except Exception as e:
            print(f'Error scanning image pairs: {e}')
            return image_pairs

    @staticmethod
    def get_image_pair_info(overlay_path: str, mask_path: str) -> Dict:
        try:
            overlay_name = Path(overlay_path).name
            mask_name = Path(mask_path).name
            prefix = overlay_name.split('_overlay')[0] if '_overlay' in overlay_name else Path(overlay_path).stem
            return {'prefix': prefix, 'overlay_name': overlay_name, 'overlay_path': overlay_path, 'mask_name': mask_name, 'mask_path': mask_path, 'is_valid': FileUtils.validate_image_pair(overlay_path, mask_path)}
        except Exception as e:
            print(f'Error reading image-pair information: {e}')
            return {}
