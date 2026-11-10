import argparse
import os
from pathlib import Path
import sys

def main():
    root = Path(__file__).resolve().parents[1]
    home = root / 'benchmark'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['train', 'evaluate'])
    p.add_argument('--model', choices=['unet', 'unetpp', 'deeplabv3plus', 'transunet', 'manet', 'panet', 'samed', 'retained'], required=True)
    p.add_argument('--data', type=Path, default=root / 'data/clinical')
    p.add_argument('--lock', type=Path, default=home / 'dataset_lock.json')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--min-delta', type=float, default=0.0)
    p.add_argument('--checkpoint-seconds', type=float, default=120.0)
    p.add_argument('--torch-threads', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--show-plot', action='store_true')
    p.add_argument('--stop-file', type=Path)
    p.add_argument('--random-init', action='store_true')
    p.add_argument('--transunet-root', type=Path, default=home / 'vendor/TransUNet')
    p.add_argument('--transunet-pretrained', type=Path, default=home / 'pretrained/R50+ViT-B_16.npz')
    args = p.parse_args()
    if args.command == 'evaluate' and args.checkpoint is None:
        p.error('evaluate requires --checkpoint')
    if args.command == 'train' and args.model == 'retained':
        p.error('Use segmentation/train.py to train the proposed model')
    sys.path.insert(0, str(home))
    os.environ['SCAR_DATASET_LOCK'] = str(args.lock.resolve())
    os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
    import benchmark as b
    b.DEFAULT_DATA = args.data.resolve()
    if args.model in ['manet', 'panet', 'samed']:
        import extended_models as extensions
        if args.model == 'panet':
            from extended_baselines.panet import adapter
            adapter.install(b)
            extensions.register_model('panet', adapter.build_model, adapter.model_spec)
        elif args.model == 'samed':
            from extended_baselines.samed import adapter
            extensions.register_model('samed', adapter.build_model, adapter.model_spec)
        original_modules = b.project_modules

        def modules():
            build, spec, *rest = original_modules()

            def extended_build(name, **kwargs):
                return extensions.build_model(name, **kwargs) if extensions.supports(name) else build(name, **kwargs)

            def extended_spec(name, **kwargs):
                return extensions.model_spec(name, **kwargs) if extensions.supports(name) else spec(name, **kwargs)
            return (extended_build, extended_spec, *rest)
        b.project_modules = modules
    args.encoder_weights = 'imagenet' if args.command == 'train' and (not args.random_init) and (args.model in ['unetpp', 'deeplabv3plus', 'manet']) else None
    if args.model != 'transunet':
        args.transunet_root = args.transunet_pretrained = None
    args.probabilities = True
    args.match_csv = None
    if args.command == 'train':
        b.train(args)
    else:
        b.evaluate(args)
if __name__ == '__main__':
    main()
