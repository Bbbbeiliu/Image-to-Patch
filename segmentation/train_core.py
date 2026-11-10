import os
import sys
import shutil
import time
import logging
import json
from tqdm.auto import tqdm
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import torch
from torch.utils.data import DataLoader
from data.custom_dataset_sam import CustomDatasetWithSAM, train_transform, val_test_transform
from models.resnet34_unet_attention import UNetWithAttentionResNet34
from models.fss_module import FSSPrototypeExtractor
from models.loss import BCEAndDiceLoss
import torch
PROJECT_DIR = PROJECT_ROOT
TRAIN_DIR = os.environ.get('SCAR_RUN_DIR', os.path.join(PROJECT_DIR, '..', 'runs', 'proposed'))
DOCTOR_IMAGE_DIR = os.path.join(PROJECT_DIR, 'data', 'doctor', 'images')
DOCTOR_MASK_DIR = os.path.join(PROJECT_DIR, 'data', 'doctor', 'masks')
PUBLIC_IMAGE_DIR = os.path.join(os.environ.get('SCAR_PUBLIC_DATA', os.path.join(PROJECT_DIR, '..', 'data', 'public')), 'images')
PUBLIC_MASK_DIR = os.path.join(os.environ.get('SCAR_PUBLIC_DATA', os.path.join(PROJECT_DIR, '..', 'data', 'public')), 'masks')
DOCTOR_SPLIT_DIR = os.environ.get('SCAR_CLINICAL_DATA', os.path.join(PROJECT_DIR, '..', 'data', 'clinical'))
PROTOTYPE_CACHE = os.path.join(TRAIN_DIR, 'prototype.pt')
BEST_MODEL_PATH = os.path.join(TRAIN_DIR, 'best.pth')
LAST_MODEL_PATH = os.path.join(TRAIN_DIR, 'last.pth')
LOG_FILE = os.path.join(TRAIN_DIR, 'training.log')
INTERRUPTED_MODEL_PATH = os.path.join(TRAIN_DIR, 'interrupted.pth')

def atomic_save(value, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + f'.{os.getpid()}.tmp'
    try:
        with open(temporary, 'wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)

def train_fss_unet_only(doctor_image_dir: str, doctor_mask_dir: str, public_image_dir: str, public_mask_dir: str, mask_suffix_doctor: str='_mask.png', mask_suffix_public: str='_segmentation.png', num_epochs: int=50, batch_size: int=4, lr: float=0.0001, lambda_fss: float=0.5, use_fss: bool=False, device: str='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    doctor_split_root = DOCTOR_SPLIT_DIR
    summary_path = os.path.join(doctor_split_root, '_audit', 'summary.json')
    if not os.path.isfile(summary_path):
        raise RuntimeError('Clinical dataset acceptance is incomplete; automatic repartitioning is disabled.')
    with open(summary_path, encoding='utf-8') as f:
        dataset_summary = json.load(f)
    if not dataset_summary.get('verification', {}).get('source_metadata_unchanged'):
        raise RuntimeError('Clinical dataset acceptance is incomplete.')
    for subset in ('train', 'val', 'test'):
        for kind in ('images', 'masks'):
            if not os.path.isdir(os.path.join(doctor_split_root, subset, kind)):
                raise FileNotFoundError('Clinical grouping directory is missing; image-level repartitioning is disabled.')
    train_dir = os.path.join(doctor_split_root, 'train')
    val_dir = os.path.join(doctor_split_root, 'val')
    train_dataset = CustomDatasetWithSAM(image_dir=os.path.join(train_dir, 'images'), mask_dir=os.path.join(train_dir, 'masks'), sam_dir=None, transform=train_transform, mask_suffix=mask_suffix_doctor)
    val_dataset = CustomDatasetWithSAM(image_dir=os.path.join(val_dir, 'images'), mask_dir=os.path.join(val_dir, 'masks'), sam_dir=None, transform=val_test_transform, mask_suffix=mask_suffix_doctor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = UNetWithAttentionResNet34(in_channels=3, out_channels=1).to(device)
    criterion = BCEAndDiceLoss(weight=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    patience = 5
    bad_epochs = 0
    public_loader = None
    fss_extractor = None
    prototype = None
    logger = logging.getLogger('fss_training')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(handler)

    def report(message):
        logger.info(message)
        tqdm.write(message)
    epoch = -1
    phase = 'initialization'
    completed_batches = 0
    best_saved = False
    try:
        report(f'START train={len(train_dataset)} val={len(val_dataset)} epochs={num_epochs} batch_size={batch_size} lr={lr} use_fss={use_fss} lambda_fss={lambda_fss} device={device}')
        report(f'Dataset: {DOCTOR_SPLIT_DIR} | Log: {LOG_FILE}')
        if use_fss:
            public_dataset = CustomDatasetWithSAM(image_dir=public_image_dir, mask_dir=public_mask_dir, sam_dir=None, transform=train_transform, mask_suffix=mask_suffix_public)
            public_loader = DataLoader(public_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            fss_extractor = FSSPrototypeExtractor(encoder=model.unet.encoder)
            report(f'Prototype support images={len(public_dataset)}, batches={len(public_loader)}')
            phase = 'prototype'
            if os.path.exists(PROTOTYPE_CACHE):
                report(f'Loading cached prototype: {PROTOTYPE_CACHE}')
                prototype = torch.load(PROTOTYPE_CACHE, map_location=device)
            else:
                report('Computing initial prototype...')
                prototype = fss_extractor.compute_global_prototype(public_loader, device, logger=logger)
                atomic_save(prototype, PROTOTYPE_CACHE)
                report(f'Prototype saved: {PROTOTYPE_CACHE}')
        for epoch in range(num_epochs):
            completed_batches = 0
            start_time = time.time()
            report(f'Epoch {epoch + 1}/{num_epochs} START')
            if use_fss and epoch % 5 == 0 and (epoch != 0):
                phase = 'prototype'
                report('Refreshing prototype...')
                prototype = fss_extractor.compute_global_prototype(public_loader, device, logger=logger)
                atomic_save(prototype, PROTOTYPE_CACHE)
            phase = 'train'
            model.train()
            running_loss = 0.0
            with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} Train', unit='batch', dynamic_ncols=True) as progress:
                for images, masks, _ in progress:
                    images, masks = (images.to(device), masks.to(device))
                    optimizer.zero_grad()
                    outputs = model(images)
                    sup_loss = criterion(outputs, masks)
                    fss_loss = torch.zeros((), device=device)
                    if use_fss:
                        fss_loss = fss_extractor.fss_loss_from_prototype(model, images, masks, prototype, criterion, device)
                    total_loss = sup_loss + lambda_fss * fss_loss
                    total_loss.backward()
                    optimizer.step()
                    completed_batches += 1
                    running_loss += total_loss.item()
                    progress.set_postfix(loss=f'{running_loss / completed_batches:.4f}', sup=f'{sup_loss.item():.4f}', fss=f'{fss_loss.item():.4f}')
                    if completed_batches % 10 == 0 or completed_batches == len(train_loader):
                        logger.info('Epoch %s Train %s/%s loss=%.6f', epoch + 1, completed_batches, len(train_loader), running_loss / completed_batches)
            avg_train_loss = running_loss / len(train_loader)
            phase = 'validation'
            completed_batches = 0
            model.eval()
            val_loss = 0.0
            with torch.no_grad(), tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} Val', unit='batch', dynamic_ncols=True) as progress:
                for images, masks, _ in progress:
                    images, masks = (images.to(device), masks.to(device))
                    val_loss += criterion(model(images), masks).item()
                    completed_batches += 1
                    progress.set_postfix(loss=f'{val_loss / completed_batches:.4f}')
                    if completed_batches % 10 == 0 or completed_batches == len(val_loader):
                        logger.info('Epoch %s Val %s/%s loss=%.6f', epoch + 1, completed_batches, len(val_loader), val_loss / completed_batches)
            avg_val_loss = val_loss / len(val_loader)
            phase = 'saving'
            if avg_val_loss < best_val_loss:
                atomic_save(model.state_dict(), BEST_MODEL_PATH)
                best_val_loss = avg_val_loss
                bad_epochs = 0
                best_saved = True
                report(f'BEST saved: {BEST_MODEL_PATH} | ValLoss={best_val_loss:.6f}')
            else:
                bad_epochs += 1
            atomic_save(model.state_dict(), LAST_MODEL_PATH)
            report(f'Epoch {epoch + 1}/{num_epochs} COMPLETE train_loss={avg_train_loss:.6f} val_loss={avg_val_loss:.6f} best_val_loss={best_val_loss:.6f} elapsed={time.time() - start_time:.1f}s | LAST saved')
            if bad_epochs >= patience:
                break
        report(f'FINISHED best_val_loss={best_val_loss:.6f}')
    except (KeyboardInterrupt, Exception) as error:
        logger.exception('STOP epoch=%s phase=%s completed_batches=%s reason=%s', epoch + 1, phase, completed_batches, type(error).__name__)
        try:
            atomic_save(model.state_dict(), INTERRUPTED_MODEL_PATH)
            report(f'Current weights saved after interruption or exception: {INTERRUPTED_MODEL_PATH}')
        except (KeyboardInterrupt, Exception):
            logger.exception('Could not save interrupted weights; previously saved best/last files remain available.')
        report(f'Best weights from this run: {BEST_MODEL_PATH}' if best_saved else 'No validation completed in this run; existing best weights were preserved.')
        raise
    finally:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
