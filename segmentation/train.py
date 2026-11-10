import argparse
import json
import os
from pathlib import Path
import sys

def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=root / 'data/clinical')
    parser.add_argument('--public-data', type=Path, default=root / 'data/public')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=None, help='Optional explicit seed for a new training run')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Choose a new output directory: ' + str(args.out))
    if not (args.data / '_audit/summary.json').is_file():
        raise FileNotFoundError('Prepare patient-level folders with tools/prepare_data.py first')
    if not all(((args.public_data / name).is_dir() for name in ['images', 'masks'])):
        raise FileNotFoundError('Public support data require images/ and masks/')
    args.out.mkdir(parents=True)
    os.environ['SCAR_RUN_DIR'] = str(args.out.resolve())
    os.environ['SCAR_CLINICAL_DATA'] = str(args.data.resolve())
    os.environ['SCAR_PUBLIC_DATA'] = str(args.public_data.resolve())
    os.environ.setdefault('TORCH_HOME', str(root / 'benchmark/torch_cache'))
    os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_core as training
    if args.seed is not None:
        import random
        import numpy as np
        import torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        training.train_transform.augment.set_random_seed(args.seed)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(optimizer='Adam', patience=5, min_delta=0.0, main_loss='BCE + Dice', prototype_weight=0.5, prototype_refresh_epochs=5, checkpoint_selection='minimum validation BCE + Dice')
    (args.out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    training.train_fss_unet_only(doctor_image_dir='', doctor_mask_dir='', public_image_dir=str(args.public_data / 'images'), public_mask_dir=str(args.public_data / 'masks'), num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, lambda_fss=0.5, use_fss=True, device=args.device)
if __name__ == '__main__':
    main()
