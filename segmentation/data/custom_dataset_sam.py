from letterbox import LetterboxTransform
import os
import json
import cv2
import numpy as np
import torch
import albumentations as A
from torch.utils.data import Dataset
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
USE_RGB = True

def _imread(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise ValueError(f'Image read failed: {path}')
    return img

def _find_mask_path(mask_dir, img_name, mask_suffix):
    stem = os.path.splitext(img_name)[0]
    candidates = [os.path.join(mask_dir, stem + mask_suffix), os.path.join(mask_dir, stem + '_segmentation.png'), os.path.join(mask_dir, stem + '.png'), os.path.join(mask_dir, stem + '.json')]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'Could not find {img_name} mask in {mask_dir} tried: ' + ', '.join((os.path.basename(c) for c in candidates)))

def _labelme_json_to_mask(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    h, w = (int(data['imageHeight']), int(data['imageWidth']))
    mask = np.zeros((h, w), dtype=np.uint8)
    for shape in data.get('shapes', []):
        pts = np.round(np.asarray(shape['points'], dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask

class CustomDatasetWithSAM(Dataset):

    def __init__(self, image_dir, mask_dir, sam_dir=None, transform=None, mask_suffix='_mask.png'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.sam_dir = sam_dir
        self.transform = transform
        self.mask_suffix = mask_suffix
        self.image_names = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXTS)])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        image_bgr = _imread(os.path.join(self.image_dir, img_name))
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) if USE_RGB else image_bgr
        mask_path = _find_mask_path(self.mask_dir, img_name, self.mask_suffix)
        if mask_path.lower().endswith('.json'):
            mask = _labelme_json_to_mask(mask_path)
            if mask.shape[:2] != image_bgr.shape[:2]:
                mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask = _imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask.shape[:2] != image_bgr.shape[:2]:
                mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        if self.sam_dir is not None:
            sam = _imread(os.path.join(self.sam_dir, os.path.splitext(img_name)[0] + '.png'), cv2.IMREAD_GRAYSCALE)
            if sam.shape[:2] != mask.shape[:2]:
                sam = cv2.resize(sam, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            sam = None
        if self.transform is not None:
            targets = {'image': image, 'mask': mask}
            if sam is not None:
                targets['sam_mask'] = sam
            augmented = self.transform(**targets)
            if sam is not None:
                sam = augmented['sam_mask']
            image = augmented['image']
            mask = augmented['mask']
        image_t = torch.tensor(np.asarray(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask_t = torch.tensor(np.asarray(mask), dtype=torch.float32).unsqueeze(0) / 255.0
        if sam is not None:
            sam_t = torch.tensor(np.asarray(sam), dtype=torch.float32).unsqueeze(0) / 255.0
            if sam_t.shape[-2:] != mask_t.shape[-2:]:
                sam_t = torch.nn.functional.interpolate(sam_t.unsqueeze(0), size=mask_t.shape[-2:], mode='nearest').squeeze(0)
        else:
            sam_t = torch.zeros_like(mask_t)
        return (image_t, mask_t, sam_t)
train_transform = LetterboxTransform(A.Compose([A.RandomRotate90(p=0.5), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomBrightnessContrast(p=0.2), A.GaussianBlur(blur_limit=3, p=0.2), A.HueSaturationValue(p=0.3)], additional_targets={'sam_mask': 'mask'}))
val_test_transform = LetterboxTransform()
